r"""
Some pytorch functions probably used in the project.
"""
import torch

class ScaleEstimatorForLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Save the input for backward pass
        ctx.save_for_backward(input)
        # Return the hard thresholded output
        input[input < 0] = 0
        input[input > 1] = 1
        return input.float()

    @staticmethod
    def backward(ctx, grad_output):
        
        input, = ctx.saved_tensors
        num_nodes = input.size(0) - (input == 0).sum(dim=0)

        input = torch.where(input != 0, 1 / input, torch.tensor(0.0))
        grad_input = grad_output / num_nodes * input
        return grad_input

class StraightThroughEstimator(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input):
        # Save the input for backward pass
        ctx.save_for_backward(input)
        # Return the hard thresholded output
        return (input > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve the saved input
        input, = ctx.saved_tensors
        # Compute the gradient of the soft threshold
        grad_input = grad_output.clone()
        # Pass the gradient through where the input was within the unit interval
        grad_input[input <= 0] = 0
        grad_input[input >= 1] = 0
        return grad_input
    
class StraightThroughEstimator_1(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        # Save the input for backward pass
        ctx.save_for_backward(input)
        input_mask = torch.argmax(input, dim=1)
        result = torch.zeros_like(input)
        result.scatter_(1, input_mask.unsqueeze(1), 1)
        # Return the hard thresholded output
        return result

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve the saved input
        input, = ctx.saved_tensors
        # Compute the gradient of the soft threshold
        grad_input = grad_output.clone()
        # Pass the gradient through where the input was within the unit interval
        grad_input[input <= 0] = 0
        grad_input[input >= 1] = 0
        return grad_input
    
class Scale(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input, scale_factor):
        ctx.scale_factor = scale_factor
        return input * scale_factor

    @staticmethod
    def backward(ctx, grad_output):
        scale_factor = ctx.scale_factor
        return grad_output * 10, None