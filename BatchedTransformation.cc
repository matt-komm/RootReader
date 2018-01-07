#include "tensorflow/core/framework/reader_base.h"
#include "tensorflow/core/framework/op.h"

#include <regex>

using namespace tensorflow;


REGISTER_OP("BatchedTransformation")
    .Input("input: float32")
    .Output("out: float32")
    .Attr("start: int")
    .Attr("size: int")
    .Attr("shape: list(int) = []")
    .Attr("transpose: bool = False")
    .SetIsStateful()
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) 
    {
        int size;
        TF_RETURN_IF_ERROR(c->GetAttr("size",&size));
        std::vector<int> shape;
        TF_RETURN_IF_ERROR(c->GetAttr("shape",&shape));
        bool transpose;
        TF_RETURN_IF_ERROR(c->GetAttr("transpose",&transpose));
        
        int batchDim = c->Value(c->Dim(c->input(0),0));
        if (shape.size()==0)
        {
            shape_inference::ShapeHandle s = c->MakeShape({batchDim,size});
            c->set_output(0,s);
        }
        else if (shape.size()==2)
        {
            if (size!=shape[0]*shape[1])
            {
                throw std::runtime_error("Mismatching slice size and target shape");
            }
            if (transpose)
            {
                shape_inference::ShapeHandle s = c->MakeShape({batchDim,shape[1],shape[0]});
                c->set_output(0,s);
            }
            else
            {
                shape_inference::ShapeHandle s = c->MakeShape({batchDim,shape[0],shape[1]});
                c->set_output(0,s);
            }
        }
        else
        {
            throw std::runtime_error("Optional shape argument needs to be of dimension 2");
        }
        return Status::OK();
    })
    .Doc(R"doc(A Reader that outputs the lines of a file delimited by '\n'.)doc");

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"


class BatchedTransformationOp:
    public OpKernel
{

    private:
        int start_;
        int size_;
        std::vector<int> shape_;
        bool transpose_;
    public:
        explicit BatchedTransformationOp(OpKernelConstruction* context): 
            OpKernel(context)
        {
            OP_REQUIRES_OK(
                context,
                context->GetAttr("start",&start_)
            );
            OP_REQUIRES_OK(
                context,
                context->GetAttr("size",&size_)
            );
            OP_REQUIRES_OK(
                context,
                context->GetAttr("shape",&shape_)
            );
            OP_REQUIRES_OK(
                context,
                context->GetAttr("transpose",&transpose_)
            );
        }
        
        virtual ~BatchedTransformationOp()
        { 
        }

        void Compute(OpKernelContext* context)
        {
            const Tensor& input_tensor = context->input(0);
            auto input = input_tensor.flat<float>();
            long num_batches = input_tensor.dim_size(0);
            long batch_size = input_tensor.dim_size(1);

            Tensor* output_tensor = nullptr;
            TensorShape shape;
            shape.AddDim(num_batches);
            if (shape_.size()==0)
            {
                shape.AddDim(size_);
            }
            else if (shape_.size()==2)
            {
                if (transpose_)
                {
                    shape.AddDim(shape_[1]);
                    shape.AddDim(shape_[0]);
                }
                else
                {
                    shape.AddDim(shape_[0]);
                    shape.AddDim(shape_[1]);
                }
            }
            OP_REQUIRES_OK(context, context->allocate_output("out", shape,&output_tensor));
            auto output = output_tensor->flat<float>();

            for (unsigned int ibatch = 0; ibatch < num_batches; ++ibatch)
            {
                if (not transpose_)
                {
                    for (unsigned int i = 0; i < size_; ++i)
                    { 
                        output(ibatch*size_+i)=input(ibatch*batch_size+start_+i);
                    }
                }
                else if (transpose_ and shape_.size()==2)
                {
                    for (unsigned int i = 0; i < shape_[0]; ++i)
                    { 
                        for (unsigned int j = 0; j < shape_[1]; ++j)
                        {
                            output(ibatch*size_+j*shape_[0]+i)=input(ibatch*batch_size+start_+i*shape_[1]+j);
                        }
                    }
                }
            }
        }  
};

REGISTER_KERNEL_BUILDER(Name("BatchedTransformation").Device(DEVICE_CPU),BatchedTransformationOp);

