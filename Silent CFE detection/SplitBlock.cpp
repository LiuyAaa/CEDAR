#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "iterator"
#include "llvm/IR/IRBuilder.h"
#include "list"
#include "iostream"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_os_ostream.h"

// #define insert_block_index 1
// #define insert_ins_index 4

using namespace llvm;

	static cl::list<std::string> Libraries("index", cl::ZeroOrMore,cl::desc("the index of block which you want split"), cl::value_desc("int"));
namespace{

  struct SplitBlock : public ModulePass{
	
    static char ID;
	SplitBlock() : ModulePass(ID){}
	int bbcount = 0;
	int inscount = 0;
	// int insert_block_index = 1;
	// std::cin>>insert_block_index;
	bool runOnModule(Module &M) override {

		auto &e = Libraries.front();
		errs()<<"parse:"<<e<<"\n";
		int insert_block_index = std::stoi(e);
		
		for(Module::iterator F = M.begin(), E = M.end(); F != E; ++F){
			for(BasicBlock &BB:F->getBasicBlockList()){
				bbcount++;

				if(bbcount == insert_block_index){
					for(Instruction &Ins:BB.getInstList()){
						inscount++;
						int len = BB.getInstList().size();
						if(inscount == len/2){
							errs()<<"number of half ins:"<<len/2<<"\n";
							//拆分基本块

							BasicBlock * oldblock = &(BB);
							Instruction * ins_point = &(Ins);
							errs()<<"split block"<<"\n";
							errs()<<oldblock->getName()<<":"<<ins_point->getName()<<"\n";
							BasicBlock * newblock = BB.splitBasicBlock(ins_point,"newblock");
							break;
						}
					}
				}
			}

		}
		errs()<<"total bb num:"<<bbcount<<"\n";
		return false;
	}
  };
}
char SplitBlock::ID = 0;
static RegisterPass<SplitBlock> X("SplitBlock","Insert Block");
