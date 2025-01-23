#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include <cstdio>
#include <llvm/IR/Constants.h>


using namespace llvm;
namespace llfi {
    class GenLLFIIndexPass : public ModulePass {
    public:
        GenLLFIIndexPass() : ModulePass(ID) {}

        virtual bool runOnModule(Module &M);

        static char ID;
    };

    char GenLLFIIndexPass::ID = 0;
    static RegisterPass<GenLLFIIndexPass> X(
            "genllfiindexpass", "Generate a unique LLFI index for each instruction",
            false, false);

    static long fi_index = 1;

    void setLLFIIndexofInst(Instruction *inst) {
        assert (fi_index >= 0 && "static instruction number exceeds index max");
        Function *func = inst->getParent()->getParent();
        LLVMContext &context = func->getContext();
        std::vector<Metadata *> llfiindex(1);
//
//        llfiindex[0] = ;
        ValueAsMetadata* metadata = ValueAsMetadata::getConstant(ConstantInt::get(Type::getInt64Ty(context), fi_index++));

        llfiindex[0] = metadata;
        MDNode *mdnode = MDNode::get(context, ArrayRef<Metadata*>(llfiindex));
        inst->setMetadata("llfi_index", mdnode);
    }

    bool GenLLFIIndexPass::runOnModule(Module &M) {
        Instruction *currinst;

        for (Module::iterator m_it = M.begin(); m_it != M.end(); ++m_it) {
            if (!m_it->isDeclaration()) {
                //m_it is a function
//                for (inst_iterator f_it = inst_begin(m_it); f_it != inst_end(m_it);
//                     ++f_it) {
//                    currinst = &(*f_it);
//                    setLLFIIndexofInst(currinst);
//                }
                for(BasicBlock& BB : m_it->getBasicBlockList()){
                    for(Instruction& I : BB.getInstList()){
                        setLLFIIndexofInst(&I);
                    }

                }


            }
        }

//        if (currinst) {
//            long totalindex = getLLFIIndexofInst(currinst);
//            FILE *outputFile = fopen("llfi.stat.totalindex.txt", "w");
//            if (outputFile)
//                fprintf(outputFile, "totalindex=%ld\n", totalindex);
//
//            fclose(outputFile);
//        }

        return true;
    }

}

