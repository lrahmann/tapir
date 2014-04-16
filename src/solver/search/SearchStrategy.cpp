#include "SearchStrategy.hpp"

#include "solver/BeliefNode.hpp"
#include "solver/BeliefTree.hpp"
#include "solver/HistoryEntry.hpp"
#include "solver/HistorySequence.hpp"
#include "solver/Solver.hpp"
#include "solver/StatePool.hpp"

#include "SearchStatus.hpp"

namespace solver {
/* ----------------------- SearchStrategy ------------------------- */
SearchStrategy::SearchStrategy(Solver *solver) :
    solver_(solver) {
}

Solver *SearchStrategy::getSolver() const {
    return solver_;
}

/* ------------------- AbstractSearchInstance --------------------- */
AbstractSearchInstance::AbstractSearchInstance(Solver *solver,
        HistorySequence *sequence, long maximumDepth) :
                solver_(solver),
                model_(solver_->getModel()),
                sequence_(sequence),
                currentNode_(sequence->getLastEntry()->getAssociatedBeliefNode()),
                currentHistoricalData_(nullptr),
                discountFactor_(model_->getDiscountFactor()),
                maximumDepth_(maximumDepth),
                status_(SearchStatus::UNINITIALIZED) {
}

SearchStatus AbstractSearchInstance::initialize() {
    status_ = initializeCustom(currentNode_);
    return status_;
}

SearchStatus AbstractSearchInstance::initializeCustom(BeliefNode */*currentNode*/) {
    return SearchStatus::INITIAL;
}

SearchStatus AbstractSearchInstance::extendSequence() {
    if (status_ != SearchStatus::INITIAL) {
        debug::show_message("WARNING: Attempted to search without initializing.");
        return status_;
    }
    HistoryEntry *currentEntry = sequence_->getLastEntry();
    status_ = SearchStatus::INITIAL;
    if (model_->isTerminal(*currentEntry->getState())) {
        debug::show_message("WARNING: Attempted to continue sequence from"
                " a terminal state.");
        status_ = SearchStatus::HIT_TERMINAL_STATE;
        return status_;
    }
    for (long currentDepth = sequence_->getStartDepth() + currentEntry->getId();; currentDepth++) {
        if (currentDepth == maximumDepth_) {
            // We have hit the depth limit.
            status_ = SearchStatus::HIT_DEPTH_LIMIT;
            break;
        }
        SearchStep step = getSearchStep();
        status_ = step.status;
        if (step.action == nullptr) {
            break;
        }
        Model::StepResult result = model_->generateStep(
                *currentEntry->getState(), *step.action);
        currentEntry->reward_ = result.reward;
        currentEntry->action_ = result.action->copy();
        currentEntry->transitionParameters_ = std::move(
                result.transitionParameters);
        currentEntry->observation_ = result.observation->copy();

        // Now we make a new history entry!
        // Add the next state to the pool
        StateInfo *nextStateInfo = solver_->getStatePool()->createOrGetInfo(
                *result.nextState);
        // Step forward in the history, and update the belief node.
        currentEntry = sequence_->addEntry(nextStateInfo);
        if (currentNode_ != nullptr && step.createNode) {
            currentNode_ = solver_->getPolicy()->createOrGetChild(
                    currentNode_, *result.action, *result.observation);
            currentHistoricalData_ = nullptr;
            currentEntry->associatedBeliefNode_ = currentNode_;
        } else {
            HistoricalData *oldData;
            if (currentNode_ == nullptr) {
                oldData = currentHistoricalData_.get();
            } else {
                oldData = currentNode_->getHistoricalData();
                currentNode_ = nullptr;
            }
            currentHistoricalData_ = oldData->createChild(
                    *result.action, *result.observation);
        }
        if (result.isTerminal) {
            status_ = SearchStatus::HIT_TERMINAL_STATE;
            break;
        }
    }
    return status_;
}

SearchStatus AbstractSearchInstance::finalize() {
    status_ = finalize();
    return status_;
}

SearchStatus AbstractSearchInstance::finalizeCustom() {
    return status_;
}

Solver *AbstractSearchInstance::getSolver() const {
    return solver_;
}

HistorySequence *AbstractSearchInstance::getSequence() const {
    return sequence_;
}

/* ------------------- AbstractSelectionInstance --------------------- */
AbstractSelectionInstance::AbstractSelectionInstance(Solver *solver,
        HistorySequence *sequence, long maximumDepth) :
        AbstractSearchInstance(solver, sequence, maximumDepth) {
}

SearchStep AbstractSelectionInstance::getSearchStep() {
    return getSearchStep(currentNode_);
}

/* ------------------- AbstractRolloutInstance --------------------- */
AbstractRolloutInstance::AbstractRolloutInstance(Solver *solver,
        HistorySequence *sequence, long maximumDepth) :
        AbstractSearchInstance(solver, sequence, maximumDepth) {
}

SearchStep AbstractRolloutInstance::getSearchStep() {
    HistoricalData *currentData;
    if (currentNode_ != nullptr) {
        currentData = currentNode_->getHistoricalData();
    } else {
        currentData = currentHistoricalData_.get();
    }
    return getSearchStep(currentData);
}

} /* namespace solver */
