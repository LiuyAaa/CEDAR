//basic implemented idea is from AddGlobalSig


#define DEBUG_TYPE "AddGlobalSig"

#include <map>
#include <set>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string.h>
#include <sys/stat.h>

#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/IntervalPartition.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/ADT/IndexedMap.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/IR/IRBuilder.h"
//#include "llvm/IR/InstIterator.h"


#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
	class AddGlobalSig : public ModulePass {
	public:
		static char ID;
		static bool flag;
		static uint32_t noOfBBs;
		bool runOnModule(Module &M);
		//static IntervalPartition* IP;
		//static LoopInfo* LI;
		/*void getAnalysisUsage(AnalysisUsage &AU) const {
          AU.addRequired<IntervalPartition>();
          AU.addRequired<LoopInfo>();
        }*/
		AddGlobalSig() : ModulePass(ID){}
	private:
		std::map<BasicBlock*, uint32_t> BBToSig; //first signature should start with 1
		//std::set<BasicBlock*> branchFanInNodes;
		// std::map<BasicBlock*, Instruction*> BBToInsertPt;
		// std::map<Function*, BasicBlock*> FunctionToExitBlock;
		// std::map<Function*, Value*> FunctionToGlobalName;
		GlobalVariable* gSig;
		// Constant* printFun;
		// Constant* gvar_ptr_stderr;

	};

	//end of class
}
//end of namespace
int stringToInt(const std::string &s) {
    int v;
    std::stringstream ss;
    ss << s;
    ss >> v;
    return v;
}
char AddGlobalSig::ID = 0;
bool AddGlobalSig::flag = false;
uint32_t AddGlobalSig::noOfBBs = 0;
//IntervalPartition* AddGlobalSig::IP = 0;
//LoopInfo* AddGlobalSig::LI = 0;

static RegisterPass<AddGlobalSig> Y("AddGlobalSig", "add global signature Implementation");

std::string getbasicblocklabel(BasicBlock* bb){
                                std::string Str;
                                raw_string_ostream OS(Str);
                              //  errs()<< i++ <<"\n";
                                bb->printAsOperand(OS, false);
                                return OS.str();
}

bool AddGlobalSig::runOnModule(Module &M){
	if(flag == true){
		return false;
	}

	//load the error function
	// Function* hookFabs = M.getFunction("find_err");
	// if(!hookFabs){
	// 	errs() << "Function fabs is not found\n";
	// 	exit(-1);
	// }


	//actual implementation goes here
	//clear and initialize all variables
	BBToSig.clear();
	//branchFanInNodes.clear();
	// BBToInsertPt.clear();
	// FunctionToExitBlock.clear();
	// FunctionToGlobalName.clear();
	gSig = NULL;
	// printFun = NULL;
	// gvar_ptr_stderr = NULL;

	//declare a global variable for global signature (gSig)
	GlobalVariable* sigVar = new GlobalVariable(M,
												IntegerType::get(M.getContext(), 32),
												false,
												GlobalValue::ExternalLinkage,
												0,
												"gSig");
	ConstantInt* const_int32_8 = ConstantInt::get(M.getContext(), APInt(32, StringRef("0"), 10));
	sigVar->setInitializer(const_int32_8);
	gSig = sigVar;

	//assign a static signature to each basic block, ingore the interval optimization for now
	for(Module::iterator I = M.begin(), E = M.end(); I != E; ++I){
		for(Function::iterator bb_iter = I->begin(); bb_iter != I->end(); ++bb_iter){
			//bb为当前基本块
			BasicBlock* bb = &(*bb_iter);
			//基本块iterator
			// BasicBlock::iterator ins_iter = bb_iter->begin();
			//找到该基本块中第一个不是phi的指令
			Instruction* Ins = bb->getFirstNonPHI();
			//获取当前基本块的label
			std::string label = getbasicblocklabel(bb);
			//去掉label中的第一个字符“%”
			std::string labelnum= label.substr(1);
			//BBToSig[bb] = ++noOfBBs;
			++noOfBBs;
			// Constant* num = ConstantInt::get(IntegerType::get(M.getContext(),32),stringToInt(labelnum));
			Constant* num = ConstantInt::get(IntegerType::get(M.getContext(),32),noOfBBs);
			// errs() << labelnum << "+" << num->getUniqueInteger() << "...." << Ins->getOpcodeName() << "\n";
			StoreInst* SI = new StoreInst(num,gSig);
			SI->insertBefore(Ins);
			//DEBUG(dbgs() << "label:"  << label << "   labelnum:" << labelnum <<"\n");
			//DEBUG(dbgs() << "BBToSig[bb]:"  << BBToSig[bb] << "\n" << "\n") ;

		}
	}

	flag = true;
	return false;
}




