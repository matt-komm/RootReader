#include "tensorflow/core/framework/reader_base.h"
#include "tensorflow/core/framework/op.h"

#include <regex>

using namespace tensorflow;


REGISTER_OP("TrainTestSplitting")
    .Attr("dtype: {int32, float}")
    .Input("num: int32") //fixed input
    .Input("input: dtype") //variable input
    .Output("train: dtype")
    .Output("test: dtype")
    .Attr("percentage: int")
    .SetIsStateful()
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) 
    {
        tensorflow::shape_inference::ShapeHandle input_shape = c->input(1);
        tensorflow::shape_inference::ShapeHandle ouput_shape = input_shape;
        c->ReplaceDim(input_shape,0,c->MakeDim(-1),&ouput_shape);
        c->set_output(0,ouput_shape);
        c->set_output(1,ouput_shape);
        
        return Status::OK();
    })
    .Doc(R"doc(A Reader that outputs the lines of a file delimited by '\n'.)doc");

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "TH1F.h"
#include "TFile.h"


class TrainTestSplittingOp:
    public OpKernel
{

    private:
        int percentage_;
        DataType dataType_;
    public:
        explicit TrainTestSplittingOp(OpKernelConstruction* context): 
            OpKernel(context)
        {
            OP_REQUIRES_OK(
                context,
                context->GetAttr("percentage",&percentage_)
            );
            OP_REQUIRES_OK(
                context,
                context->GetAttr("dtype",&dataType_)
            );
        }
        
        
        virtual ~TrainTestSplittingOp()
        { 
        }
        
        static long hash(long value) //unfortunately std::hash<int> maps back
        {
            constexpr static uint32_t c2=0x27d4eb2d; // a prime or an odd constant
            uint32_t key = (value ^ 61) ^ (value >> 16);
            key = key + (key << 3);
            key = key ^ (key >> 4);
            key = key * c2;
            key = key ^ (key >> 15);
            return key;
        }
        
        bool isTesting(int value) const
        {
            //use hash for determinstic randomization
            return hash(value)%100<percentage_;
        }
        
        void Compute(OpKernelContext* context)
        {
            if (dataType_==DT_FLOAT)
            {
                ComputeTmpl<float>(context);
            }
            else if (dataType_==DT_INT32)
            {
                ComputeTmpl<int>(context);
            }
            else
            {
                throw std::runtime_error("Data type '"+DataTypeString(dataType_)+"' not (yet) supported");
            }
        }
        
        template<class T> void ComputeTmpl(OpKernelContext* context)
        {
            const Tensor& num_tensor = context->input(0);
            auto nums = num_tensor.flat<int>();
            
            const Tensor& data_tensor = context->input(1);
            auto data = data_tensor.flat<T>();
            
            long input_batch_length = data_tensor.dim_size(0);
            if (input_batch_length!=num_tensor.dim_size(0))
            {
                throw std::runtime_error("Input ("+
                    std::to_string(input_batch_length)+
                    ") and num tensor ("+
                    std::to_string(num_tensor.dim_size(0))+
                    ") need to be of same first dimension");
            }
            
            int train_batch_length = 0;
            int test_batch_length = 0;
            
            for (unsigned int ibatch = 0; ibatch < input_batch_length; ++ibatch)
            {
                if (isTesting(nums(ibatch)))
                {
                    test_batch_length++;
                }
                else
                {
                    train_batch_length++;
                }
            }
            
            Tensor* train_output_tensor = nullptr;
            TensorShape train_output_shape = data_tensor.shape();
            train_output_shape.set_dim(0,train_batch_length);
            OP_REQUIRES_OK(context, context->allocate_output("train", train_output_shape,&train_output_tensor));
            auto train_data = train_output_tensor->flat<T>();
            
            Tensor* test_output_tensor = nullptr;
            TensorShape test_output_shape = data_tensor.shape();
            test_output_shape.set_dim(0,test_batch_length);
            OP_REQUIRES_OK(context, context->allocate_output("test", test_output_shape,&test_output_tensor));
            auto test_data = test_output_tensor->flat<T>();

            int itrain_batch = 0;
            int itest_batch = 0;
            int64_t elems_per_batch = data_tensor.NumElements()/input_batch_length;
            for (unsigned int ibatch = 0; ibatch < input_batch_length; ++ibatch)
            {
                if (isTesting(nums(ibatch)))
                {
                    for (unsigned int ielem = 0; ielem < elems_per_batch; ++ielem)
                    {
                        test_data(itest_batch*elems_per_batch+ielem)=data(ibatch*elems_per_batch+ielem);
                    }
                    itest_batch++;
                }
                else
                {
                    for (unsigned int ielem = 0; ielem < elems_per_batch; ++ielem)
                    {
                        train_data(itrain_batch*elems_per_batch+ielem)=data(ibatch*elems_per_batch+ielem);
                    }
                    itrain_batch++;
                }
            }
        }  
};

REGISTER_KERNEL_BUILDER(Name("TrainTestSplitting").Device(DEVICE_CPU),TrainTestSplittingOp);

