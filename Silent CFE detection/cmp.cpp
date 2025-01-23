#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/ArrayRef.h"

#include <iostream>
#include <fstream>
#include <set>
#include <list>
#include <sstream>
#include <algorithm>
#include <sstream>

using namespace llvm;

static cl::opt<std::string> cl_select_set_file_name("select_set_file", cl::desc("Specify the file of selected sets"), cl::value_desc("file name"));

namespace {
    struct test : public ModulePass {
        static char ID;
        std::vector<long> selectedIndexInsts;
        std::vector<long> duplicatedIndexInsts;

        std::vector<Instruction*> selectedInsts, duplicatedInsts;
        Module *appModule;
        
        Function *hook;
        GlobalVariable *brConditionGlobalVar;

        test() : ModulePass(ID) {}

        virtual bool runOnModule(Module &M) {
            Constant *hookFunc;
            hookFunc = M.getOrInsertFunction("dumpIndex", Type::getVoidTy(M.getContext()), Type::getInt64Ty(M.getContext()));

            hook = cast<Function>(hookFunc);
            appModule = &M;
            readSelectSet();

            // Create a global variable to store BR condition
            brConditionGlobalVar = new GlobalVariable(M, Type::getInt1Ty(M.getContext()), false, GlobalValue::InternalLinkage, ConstantInt::get(Type::getInt1Ty(M.getContext()), 0), "br_condition");

            for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
                for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB) {
                    for (BasicBlock::iterator BI = BB->begin(), BE = BB->end(); BI != BE; ++BI) {
                        long llfiIndex = getLLFIIndexofInst(BI.operator->());
                        errs() << llfiIndex << "\n";
                        if (std::find(selectedIndexInsts.begin(), selectedIndexInsts.end(), llfiIndex) != selectedIndexInsts.end()) {
                            if (std::find(duplicatedIndexInsts.begin(), duplicatedIndexInsts.end(), llfiIndex) != duplicatedIndexInsts.end()) {
                                continue;
                            }

                            int opcode = BI->getOpcode();
                            if (isa<StoreInst>(BI) || isa<TerminatorInst>(BI) || opcode >= Instruction::ICmp || isa<AllocaInst>(BI)) {
                                errs() << llfiIndex << "\n";
                                continue;
                            }

                            selectedInsts.push_back(BI.operator->());

                            Instruction* duplicatedInst = BI->clone();

                            for (unsigned int i = 0; i < duplicatedInst->getNumOperands(); i++) {
                                duplicatedInst->setOperand(i, BI->getOperand(i));
                            }

                            BasicBlock::iterator nextInst = BI;
                            nextInst++;
                            duplicatedInst->insertBefore(nextInst.operator->());
                            duplicatedIndexInsts.push_back(llfiIndex);
                            duplicatedInsts.push_back(duplicatedInst);

                        } else {
                            continue;
                        }
                    }
                }
            }

            for (int i = 0; i < selectedInsts.size(); i++) {
                Instruction* currentInst = selectedInsts.at(i);
                for (unsigned int j = 0; j < currentInst->getNumOperands(); j++) {
                    if (getIndexOfDependentInst(currentInst->getOperand(j)) >= 0) {
                        long dependentIndex = getIndexOfDependentInst(currentInst->getOperand(j));
                        Instruction* currentDuplicatedInst = getInstFromDuplicationByIndex(getLLFIIndexofInst(currentInst));
                        Instruction* dependentDuplicationInst = getInstFromDuplicationByIndex(dependentIndex);
                        currentDuplicatedInst->setOperand(j, dependentDuplicationInst);
                    }
                }
            }

            for (int i = 0; i < selectedInsts.size(); i++) {
                Instruction* currentInst = selectedInsts.at(i);
                if (!isUsedInOtherOperand(currentInst)) {
                    Instruction* duplicatedInst = getInstFromDuplicationByIndex(getLLFIIndexofInst(currentInst));
                    BasicBlock::iterator cmpPosInst = duplicatedInst->getIterator();
                    cmpPosInst++;
                    if (cmpPosInst->getOpcode() == Instruction::BitCast) {
                        continue;
                    }
                    Instruction* cmpInst = insertCmp(cmpPosInst.operator->(), currentInst, duplicatedInst);

                    BasicBlock::iterator checkPosInst = cmpPosInst;
                    insertCheckFunction(checkPosInst.operator->(), cmpInst, M.getContext());
                }
            }

            // Handle BR instructions
            for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
                for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB) {
                    for (BasicBlock::iterator BI = BB->begin(), BE = BB->end(); BI != BE; ++BI) {
                        if (isa<BranchInst>(BI)) {
                            BranchInst *brInst = cast<BranchInst>(BI);
                            if (brInst->isConditional()) {
                                Value *condition = brInst->getCondition();
                                IRBuilder<> builder(brInst);
                                builder.CreateStore(condition, brConditionGlobalVar);

                                for (unsigned int i = 0; i < brInst->getNumSuccessors(); i++) {
                                    BasicBlock *successorBB = brInst->getSuccessor(i);
                                    Instruction *firstInst = &*successorBB->getFirstInsertionPt();
                                    IRBuilder<> successorBuilder(firstInst);
                                    LoadInst *loadedCondition = successorBuilder.CreateLoad(Type::getInt1Ty(M.getContext()), brConditionGlobalVar, "loaded_condition");
                                    CmpInst *cmpInst = successorBuilder.CreateICmpEQ(condition, loadedCondition, "br_condition_cmp");
                                    insertCheckFunction(firstInst, cmpInst, M.getContext());
                                }
                            }
                        }
                    }
                }
            }

            errs() << "Added Duplication " << duplicatedInsts.size() << '\n';
            return false;
        }

        void insertCheckFunction(Instruction* instPos, Instruction* cmpInst, LLVMContext &context) {
            std::vector<Value*> checker_args_vector(1);
            checker_args_vector[0] = cmpInst;
            ArrayRef<llvm::Value*> checker_args(checker_args_vector);
            std::vector<llvm::Type*> checker_arg_types_vector(1);
            checker_arg_types_vector[0] = cmpInst->getType();
            ArrayRef<llvm::Type*> checker_arg_types(checker_arg_types_vector);

            FunctionType* checker_type = FunctionType::get(Type::getVoidTy(context), checker_arg_types, false);
            Constant* checker_handler_c = appModule->getOrInsertFunction("check_df", checker_type);
            Function* checker_handler = dyn_cast<Function>(checker_handler_c);
            CallInst::Create(checker_handler, checker_args, "", instPos);
        }

        Instruction* insertCmp(Instruction* insertionLocationInst, Instruction* cmpInstOne, Instruction* cmpInstTwo) {
            CmpInst* check_cmp = NULL;
            if (cmpInstOne->getType()->isFloatingPointTy()) {
                check_cmp = CmpInst::Create(Instruction::FCmp, CmpInst::CmpInst::FCMP_UEQ, cmpInstOne, cmpInstTwo, "check_cmp", insertionLocationInst);
            } else {
                check_cmp = CmpInst::Create(Instruction::ICmp, CmpInst::ICMP_EQ, cmpInstOne, cmpInstTwo, "check_cmp", insertionLocationInst);
            }
            return check_cmp;
        }

        long getLLFIIndexofInst(Instruction *inst) {
            MDNode *mdnode = inst->getMetadata("llfi_index");
            if (mdnode) {
                Metadata* mdata = mdnode->getOperand(0).operator->();
                ValueAsMetadata* v = dyn_cast<ValueAsMetadata>(mdata);
                ConstantInt *cns_index = dyn_cast<ConstantInt>(v->getValue());
                return cns_index->getSExtValue();
            }
            return 0;
        }

        void readSelectSet() {
            std::ifstream select_set_file;
            select_set_file.open(cl_select_set_file_name.c_str());
            if (!select_set_file.is_open()) {
                errs() << "\nERROR: can not open select_set_file!\n";
                exit(1);
            }

            while (select_set_file.good()) {
                std::string line;
                getline(select_set_file, line);
                if (line.empty()) continue;
                else {
                    long currentIndex = atol(line.c_str());
                    errs() << "Select Index: " << currentIndex << "\n";
                    selectedIndexInsts.push_back(currentIndex);
                }
            }
        }

        long getIndexOfDependentInst(Value* operand) {
            if (!isa<Instruction>(operand)) return -1;

            Instruction* operandInst = cast<Instruction>(operand);

            for (int i = 0; i < selectedInsts.size(); i++) {
                if (operandInst == selectedInsts.at(i)) return getLLFIIndexofInst(selectedInsts.at(i));
            }
            return -1;
        }

        Instruction* getInstFromDuplicationByIndex(long llfiIndex) {
            for (int i = 0; i < duplicatedInsts.size(); i++) {
                if (llfiIndex == getLLFIIndexofInst(duplicatedInsts.at(i))) {
                    return duplicatedInsts.at(i);
                }
            }
            return nullptr;
        }

        bool isUsedInOtherOperand(Instruction* targetInst) {
            for (int i = 0; i < selectedInsts.size(); i++) {
                Instruction* currentInst = selectedInsts.at(i);
                for (unsigned int j = 0; j < currentInst->getNumOperands(); j++) {
                    Value* operandValue = currentInst->getOperand(j);
                    if (isa<Instruction>(operandValue)) {
                        Instruction* operandInst = cast<Instruction>(operandValue);
                        if (operandInst == targetInst) return true;
                    }
                }
            }
            return false;
        }
    };
}
char test::ID = 0;
static RegisterPass<test> X("cmp", "test function exist", false, false);