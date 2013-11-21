#include <climits>

#include "Histories.h"

using namespace std;

Histories::Histories() {
}

Histories::~Histories() {
    vector<HistorySequence*>::iterator it;
    for (it = allHistSeq.begin(); it != allHistSeq.end(); it++) {
        delete (*it);
    }
    allHistSeq.resize(0);
}

void Histories::readHistories(ifstream &inFile, StatePool *stPool) {
    string tmpStr;
    getline(inFile, tmpStr);

    while (tmpStr.find("HISTORIES-BEGIN") == string::npos) {
        getline(inFile, tmpStr);
    }
    getline(inFile, tmpStr);

    long seqId, entryId, stId;
    StateWrapper *stPtr;
    pair<set<StateWrapper*, CompStVals>::iterator, bool> ret;
    while (tmpStr.find("HISTORIES-END") == string::npos) {
        stringstream sstr(tmpStr);
        string usrStr;
        sstr >> usrStr >> usrStr >> seqId >> entryId >> usrStr >> usrStr
                >> stId;
        stPtr = stPool->getStPtr(stId);
        HistoryEntry* histEntry = new HistoryEntry(seqId, entryId, stPtr, sstr);
        stPtr->addInfo(histEntry);
        if (entryId == 0) {
            HistorySequence* histSeq = new HistorySequence(histEntry);
            allHistSeq.push_back(histSeq);
        } else {
            allHistSeq[seqId]->addEntry(histEntry);
        }
        getline(inFile, tmpStr);
    }
}

void Histories::write(ostream &os) {
    vector<HistorySequence*>::iterator it;
    //cerr << "#histSeq: " << allHistSeq.size() << endl;
    for (it = allHistSeq.begin(); it != allHistSeq.end(); it++) {
        (*it)->write(os);
    }
}
