#ifndef TENSORFLOWROOT_BRANCH_H_
#define TENSORFLOWROOT_BRANCH_H_

template<typename IN, typename OUT> class SingleBranch;

template<typename OUT>
class Branch
{
    protected:
        string name_;
    public:
        Branch(const string& name):
            name_(name)
        {
        }
        
        virtual ~Branch()
        {
        }
        
        static tensorflow::Status createFromConfig(
            const std::string& config,
            std::vector<std::shared_ptr<Branch<OUT>>>& branches, 
            std::vector<std::shared_ptr<SingleBranch<unsigned int, OUT>>>& array_length_branches
        );
        
        template<typename IN> static std::shared_ptr<Branch<OUT>> createBranchByType(
            const std::string& branch_name,
            std::shared_ptr<SingleBranch<unsigned int, OUT>>& array_length_branch,
            unsigned int multiplicity_max
        );
        
        inline const string& name() const
        {
            return name_;
        }
        
        static OUT resetNanOrInf(const OUT& v, const OUT& reset)
        {
            if (std::isnan(v) or std::isinf(v))
            {
                return reset;
            }
            return v;
        }
        
        virtual const std::type_info& getInputType() const = 0;
        virtual const std::type_info& getOutputType() const
        {
            return typeid(OUT);
        }
        
        virtual unsigned int multiplicity() const = 0;
        virtual void setBranchAddress(TTree* tree) = 0;
        virtual unsigned int fillTensor(typename TTypes<OUT>::Flat& flatTensor, unsigned int index, const OUT& reset) const = 0;
};

template<typename IN, typename OUT=IN>
class SingleBranch:
    public Branch<OUT>
{
    private:
        IN value_;
    public:
        SingleBranch(const string& name):
            Branch<OUT>(name)
        {
        }
        
        virtual ~SingleBranch()
        {
        }
        
        inline const IN& value() const
        {
            return value_;
        }
        
        virtual const std::type_info& getInputType() const
        {
            return typeid(IN);
        }
        
        virtual void setBranchAddress(TTree* tree)
        {
            if(tree->SetBranchAddress(Branch<OUT>::name().c_str(),&value_)<0)
            {
                throw std::runtime_error("No branch with name '"+Branch<OUT>::name()+"' in tree");
            }
        }
        
        virtual unsigned int fillTensor(typename TTypes<OUT>::Flat& flatTensor,unsigned int index, const OUT& reset) const
        {
            flatTensor(index)=Branch<OUT>::resetNanOrInf(value_,reset);
            //std::cout<<index<<": "<<Branch<OUT>::name()<<"="<<flatTensor(index)<<std::endl;
            return index+1;
        }
        
        virtual unsigned int multiplicity() const
        {
            return 1;
        }
}; 

template<typename IN, typename OUT=IN,size_t N=50>
class ArrayBranch:
    public Branch<OUT>
{
    private:
        alignas(16) IN values_[N]; //there is some odd bug when using dynamic allocated arrays and root
        unsigned int size_;
        std::shared_ptr<SingleBranch<unsigned int,OUT>> length_;
        
    public:
        ArrayBranch(const string& name, std::shared_ptr<SingleBranch<unsigned int,OUT>>& length, unsigned int size):
            Branch<OUT>(name),
            length_(length),
            size_(size)
        {
            if (size_>N)
            {
                throw std::runtime_error("Internal buffer ("+std::to_string(N)+") to small for root array ("+std::to_string(size)+").");
            }
        }
        
        inline const IN& value(unsigned int index) const
        {
            if (index>=size_)
            {
                throw std::runtime_error("Array index out-of-range");
            }
            return values_[index];
        }
        
        virtual ~ArrayBranch()
        {
        }
        
        //error codes: https://root.cern.ch/doc/master/classTTree.html#a1a48bf75621868a514741b27252cad96
        virtual void setBranchAddress(TTree* tree)
        {
            if(tree->SetBranchAddress(Branch<OUT>::name().c_str(),values_)<0)
            {
                throw std::runtime_error("No branch with name '"+Branch<OUT>::name()+"' in tree");
            }
        }
        
        virtual unsigned int fillTensor(typename TTypes<OUT>::Flat& flatTensor, unsigned int index, const OUT& reset) const
        {
            //std::cout<<Branch<OUT>::name()<<", length="<<length_->value()<<std::endl;
            for (unsigned int i = 0; i < std::min(length_->value(),size_); ++i)
            {
                //std::cout<<(index+i)<<": "<<values_[i]<<std::endl;
                flatTensor(index+i)=Branch<OUT>::resetNanOrInf(values_[i],reset);
            }
            for (unsigned int i = std::min(length_->value(),size_); i < size_; ++i)
            {
                //std::cout<<(index+i)<<": padded"<<std::endl;
                flatTensor(index+i) = 0; //zero padding
            }
            
            return index+size_;
        }
        
        virtual const std::type_info& getInputType() const
        {
            return typeid(IN);
        }
        
        virtual unsigned int multiplicity() const
        {
            return size_;
        }
};


template<typename OUT>
tensorflow::Status Branch<OUT>::createFromConfig(
    const std::string& config,
    std::vector<std::shared_ptr<Branch<OUT>>>& branches, 
    std::vector<std::shared_ptr<SingleBranch<unsigned int, OUT>>>& array_length_branches
)
{
    std::string branch_name = config;
    char type_name = 'F';
    std::string multiplicty_name = "";
    unsigned int multiplicity_max = 1;
    std::shared_ptr<SingleBranch<unsigned int, OUT>> array_length_branch(nullptr);
    
    auto multiplicity_delimiter_begin = std::find(branch_name.begin(),branch_name.end(),'[');
    auto multiplicity_delimiter_middle = std::find(multiplicity_delimiter_begin,branch_name.end(),',');
    auto multiplicity_delimiter_end = std::find(multiplicity_delimiter_middle,branch_name.end(),']');
    if (multiplicity_delimiter_begin!=branch_name.end() and multiplicity_delimiter_end!=branch_name.end())
    {
        if (multiplicity_delimiter_begin<multiplicity_delimiter_middle and
            multiplicity_delimiter_middle<multiplicity_delimiter_end
        )
        {
            try
            {
                multiplicity_max = std::stol(std::string(multiplicity_delimiter_middle+1,multiplicity_delimiter_end));
            }
            catch (std::exception e)
            {
                return tensorflow::Status(
                    tensorflow::error::INVALID_ARGUMENT,
                    "Cannot parse config syntax. Array multiplicity not an integer: "+
                        std::string(multiplicity_delimiter_middle+1,multiplicity_delimiter_end)
                    );
            }
            multiplicty_name = std::string(multiplicity_delimiter_begin+1,multiplicity_delimiter_middle);
        }
        else
        {
            return tensorflow::Status(
                tensorflow::error::INVALID_ARGUMENT,
                "Cannot parse config syntax with array config: "+
                    std::string(multiplicity_delimiter_begin,multiplicity_delimiter_end+1)
                );
        }
        branch_name = std::string(branch_name.begin(),multiplicity_delimiter_begin)+
                      std::string(multiplicity_delimiter_end+1,branch_name.end());
        
        auto find_branch_it = std::find_if(
            array_length_branches.begin(),
            array_length_branches.end(),
            [&branch_name](const std::shared_ptr<SingleBranch<unsigned int,OUT>>& elem) -> bool
            {
                return elem->name()==branch_name;
            }
        );
        if (find_branch_it==array_length_branches.end())
        {
            array_length_branch.reset(
                new SingleBranch<unsigned int,OUT>(
                    multiplicty_name
                )
            );
            array_length_branches.push_back(
                array_length_branch
            );
        }
        else
        {
            array_length_branch = *find_branch_it;
        }

    }
    
    auto type_delimiter = std::find(branch_name.begin(),branch_name.end(),'/');
    if (type_delimiter!=branch_name.end())
    {
        if ((type_delimiter+2)==branch_name.end())
        {
            type_name = *(type_delimiter+1);
            branch_name = std::string(branch_name.begin(),type_delimiter);
        }
        else
        {
            return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,"Cannot parse config syntax with type config: "+std::string(type_delimiter+1,branch_name.end()));
        }
    }   
    
    /*
    https://root.cern.ch/doc/master/classTTree.html#ac1fa9466ce018d4aa739b357f981c615
    B : an 8 bit signed integer (Char_t)
    b : an 8 bit unsigned integer (UChar_t)
    S : a 16 bit signed integer (Short_t)
    s : a 16 bit unsigned integer (UShort_t)
    I : a 32 bit signed integer (Int_t)
    i : a 32 bit unsigned integer (UInt_t)
    F : a 32 bit floating point (Float_t)
    D : a 64 bit floating point (Double_t)
    L : a 64 bit signed integer (Long64_t)
    l : a 64 bit unsigned integer (ULong64_t)
    O : [the letter o, not a zero] a boolean (Bool_t)
    */
    switch (type_name)
    {
        case 'B': 
            branches.push_back(createBranchByType<Char_t>(
                branch_name,
                array_length_branch,
                multiplicity_max
            ));
            return tensorflow::Status::OK();
        case 'b': 
            branches.push_back(createBranchByType<UChar_t>(
                branch_name,
                array_length_branch,
                multiplicity_max
            ));
            return tensorflow::Status::OK();
        case 'S': 
            branches.push_back(createBranchByType<Short_t>(
                branch_name,
                array_length_branch,
                multiplicity_max
            ));
            return tensorflow::Status::OK();
        case 's': 
            branches.push_back(createBranchByType<UShort_t>(
                branch_name,
                array_length_branch,
                multiplicity_max
            ));
            return tensorflow::Status::OK();
        case 'I': 
            branches.push_back(createBranchByType<Int_t>(
                branch_name,
                array_length_branch,
                multiplicity_max
            ));
            return tensorflow::Status::OK();
        case 'i': 
            branches.push_back(createBranchByType<UInt_t>(
                branch_name,
                array_length_branch,
                multiplicity_max
            ));
            return tensorflow::Status::OK();
        case 'F': 
            branches.push_back(createBranchByType<Float_t>(
                branch_name,
                array_length_branch,
                multiplicity_max
            ));
            return tensorflow::Status::OK();
        case 'D': 
            branches.push_back(createBranchByType<Double_t>(
                branch_name,
                array_length_branch,
                multiplicity_max
            ));
            return tensorflow::Status::OK();
        case 'L': 
            branches.push_back(createBranchByType<Long64_t>(
                branch_name,
                array_length_branch,
                multiplicity_max
            ));
            return tensorflow::Status::OK();
        case 'l': 
            branches.push_back(createBranchByType<ULong64_t>(
                branch_name,
                array_length_branch,
                multiplicity_max
            ));
            return tensorflow::Status::OK();
        case 'o': 
            branches.push_back(createBranchByType<Bool_t>(
                branch_name,
                array_length_branch,
                multiplicity_max
            ));
            return tensorflow::Status::OK();
        default:
                return tensorflow::Status(
                    tensorflow::error::INVALID_ARGUMENT,
                    "Cannot parse config syntax with type specifier: "+
                        type_name
                );
    }
    
    return tensorflow::Status(
        tensorflow::error::INVALID_ARGUMENT,
        "Cannot parse config syntax: "+config
    );
           
    
}

template<typename OUT> template<typename IN> std::shared_ptr<Branch<OUT>> Branch<OUT>::createBranchByType(
    const std::string& branch_name,
    std::shared_ptr<SingleBranch<unsigned int, OUT>>& array_length_branch,
    unsigned int multiplicity_max
)
{
    if (array_length_branch)
    {
        return std::make_shared<ArrayBranch<IN,OUT>>(
            branch_name,
            array_length_branch,
            multiplicity_max
        );
    }
    else
    {
        return std::make_shared<SingleBranch<IN,OUT>>(
            branch_name
        );
    }
}

#endif
