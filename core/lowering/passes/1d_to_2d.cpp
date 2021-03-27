#include <memory>
#include "torch/csrc/jit/ir/ir.h"
#include "torch/csrc/jit/ir/subgraph_matcher.h"
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include "core/util/prelude.h"


namespace trtorch {
namespace core {
namespace lowering {  
namespace passes { 

namespace {
  using namespace torch::jit;
  std::string getFuncName(Value* func_value) {
    auto func = func_value->type()->expectRef<FunctionType>().function();
    const auto& qname = func->qualname();
    const auto& name = qname.qualifiedName();
    auto rdot_idx = name.rfind('.');
    if (rdot_idx != std::string::npos) {
      return name.substr(rdot_idx + 1, name.length());
    } else {
      return name;
    }
  }
  
  Value* getValue(
		  const std::string& name,
		  const std::unordered_map<const Value*, Value*>& match_vmap,
		  const std::unordered_map<std::string, Value*>& vmap) {
    return match_vmap.at(vmap.at(name));
  }
  
  c10::optional<IValue> getIValue(
				  const std::string& name,
				  const std::unordered_map<const Value*, Value*>& match_vmap,
				  const std::unordered_map<std::string, Value*>& vmap) {
    return toIValue(getValue(name, match_vmap, vmap));
  }
  
  std::unordered_map<std::string, c10::IValue> getConvParams(
							     const Match& match,
							     const std::unordered_map<std::string, Value*>& vmap) {
    std::unordered_map<std::string, c10::IValue> calc_values;
    const auto& match_vmap = match.values_map;
    auto transposed_value = getIValue("transposed", match_vmap, vmap).value();
    calc_values["transposed"] = transposed_value;
    auto output_padding_value =
      getIValue("output_padding", match_vmap, vmap).value();
    calc_values["output_padding"] = output_padding_value;
    auto stride_value = getIValue("stride", match_vmap, vmap).value();
    calc_values["stride"] = stride_value;
    auto padding_value = getIValue("padding", match_vmap, vmap).value();
    calc_values["padding"] = padding_value;
    auto dilation_value = getIValue("dilation", match_vmap, vmap).value();
    calc_values["dilation"] = dilation_value;
    return calc_values;
  }
  
  void replaceConvolutionWithAtenConv1d(std::shared_ptr<Graph>& graph) {
    std::string convolution = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[],
          %transposed:bool, %output_padding:int[], %groups:int, %benchmark:bool,
          %deterministic:bool, %cudnn_enabled:bool, %allow_tf32:bool):
        %r = aten::_convolution(%a, %w, %b, %stride, %padding, %dilation,
            %transposed, %output_padding, %groups, %benchmark, %deterministic, %cudnn_enabled, %allow_tf32)
        return (%r) )";  
    std::string conv1d = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[],
          %transposed:bool, %output_padding:int[], %groups:int, %benchmark:bool,
          %deterministic:bool, %cudnn_enabled:bool, %allow_tf32:bool):
        %r = aten::conv1d(%a, %w, %b, %stride, %padding, %dilation, %groups)
        return (%r) )";
    std::string conv_transpose1d = R"(
      graph(%a, %w, %b, %stride:int[], %padding:int[], %dilation:int[],
          %transposed:bool, %output_padding:int[], %groups:int, %benchmark:bool,
          %deterministic:bool, %cudnn_enabled:bool, %allow_tf32:bool):
        %r = aten::conv_transpose1d(%a, %w, %b, %stride, %padding, %output_padding, %groups, %dilation)
        return (%r) )";
    
    // Filter the unsupported case
    auto filter_conv1d = [](const Match& match,
			    const std::unordered_map<std::string, Value*>& vmap) {
      auto calc_value_map = getConvParams(match, vmap);
      if (calc_value_map["output_padding"].toIntList().size() != 1 ||
	  calc_value_map["stride"].toIntList().size() != 1 ||
	  calc_value_map["padding"].toIntList().size() != 1 ||
	  calc_value_map["dilation"].toIntList().size() != 1) {
	return false;
      }
      return !calc_value_map["transposed"].toBool();
    };
    auto filter_conv_transpose1d =
      [](const Match& match,
	 const std::unordered_map<std::string, Value*>& vmap) {
      auto calc_value_map = getConvParams(match, vmap);
      if (calc_value_map["output_padding"].toIntList().size() != 1 ||
	  calc_value_map["stride"].toIntList().size() != 1 ||
	  calc_value_map["padding"].toIntList().size() != 1 ||
	  calc_value_map["dilation"].toIntList().size() != 1) {
	return false;
      }
      return calc_value_map["transposed"].toBool();
    };
	  
    SubgraphRewriter rewriter_conv1d;
    rewriter_conv1d.RegisterRewritePattern(convolution, conv1d);
      rewriter_conv1d.runOnGraph(graph, filter_conv1d);

    SubgraphRewriter rewriter_conv_transpose1d;
    rewriter_conv_transpose1d.RegisterRewritePattern(
						     convolution, conv_transpose1d);
    rewriter_conv_transpose1d.runOnGraph(graph, filter_conv_transpose1d);
  }
  
  void replaceConv1dWithConv2d(std::shared_ptr<torch::jit::Graph>& graph) {
      std::string conv_1d_pattern = R"IR(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %r = aten::conv1d(%input, %weight, %bias, %stride, %padding, %dilation, %groups)
        return (%r) )IR";

      std::string conv_2d_pattern = R"IR(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %zero : int = prim::Constant[value=0]()
        %one : int = prim::Constant[value=1]()
        %two : int = prim::Constant[value=2]()

        %stride_2d : int[] = prim::ListConstruct(%one, %stride)
        %padding_2d : int[] = prim::ListConstruct(%zero, %padding)
        %dilation_2d : int[] = prim::ListConstruct(%one, %dilation)

        %dim0 : int = aten::sizes(%input)
        %dim_2d : int[] = prim::ListConstruct(%dim0, %one)
        %input_2d : Tensor = aten::view(%input, %dim_2d)
        %wdim0 : int = aten::sizes(%weight)
        %wdim_2d : int[] = prim::ListConstruct(%wdim0, %one)
        %weight_2d : Tensor = aten::view(%weight, %wdim_2d)

        %output_2d = aten::conv2d(
            %input_2d, %weight_2d, %bias, %stride_2d, %padding_2d, %dilation_2d, %groups)
        %odim0 : int = aten::size(%output_2d, %zero)
        %odim1 : int = aten::size(%output_2d, %one)
        %odim2 : int = aten::size(%output_2d, %two)
        %odim  : int[] = prim::ListConstruct(%odim0, %odim1, %odim2) 
        %output : Tensor = aten::view(%output_2d, %odim)
        return (%output) )IR";

      torch::jit::SubgraphRewriter rewriter;
      rewriter.RegisterRewritePattern(conv_1d_pattern, conv_2d_pattern);
      rewriter.runOnGraph(graph);
      // TODO: conv_transpose1D->2D
  }  
}

  void AdaptivePooling1DToPooling2D(std::shared_ptr<Graph>& graph) {
    std::string pooling_1d_pattern = R"IR(
    graph(%input, %size:int):
        %r = aten::adaptive_avg_pool1d(%input, %size)
        return (%r) )IR";
    
    std::string pooling_2d_pattern = R"IR(
    graph(%input, %size:int):
        %one : int = prim::Constant[value=1]()
        %three : int = prim::Constant[value=3]()
        %input_2d : Tensor = aten::unsqueeze(%input, %three)
        %size_2d : int[] = prim::ListConstruct(%size, %one)
        %r_2d = aten::adaptive_avg_pool2d(%input, %size)
        %r : Tensor = aten::squeeze(%r_2d, %one)
        return (%r) )IR";
      torch::jit::SubgraphRewriter rewriter;
      rewriter.RegisterRewritePattern(pooling_1d_pattern, pooling_2d_pattern);
      rewriter.runOnGraph(graph);
      LOG_GRAPH("Post map conv1d -> conv2d: " << *graph);
      // TODO: other types of pooling
  }
  
  void Conv1DToConv2D(std::shared_ptr<torch::jit::Graph>& graph) {
    // Replace _convolution with conv1d and conv2d
    replaceConvolutionWithAtenConv1d(graph);
    replaceConv1dWithConv2d(graph);
    AdaptivePooling1DToPooling2D(graph);
    LOG_GRAPH("Post map conv1d -> conv2d: " << *graph);
  }


} // namespace passes
} // namespace lowering
} // namespace core
} // namespace trtorch
