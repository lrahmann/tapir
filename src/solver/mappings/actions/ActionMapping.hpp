#ifndef SOLVER_ACTIONMAPPING_HPP_
#define SOLVER_ACTIONMAPPING_HPP_

#include "global.hpp"

#include "solver/abstract-problem/Action.hpp"

#include "solver/mappings/actions/ActionMappingEntry.hpp"

namespace solver {
class ActionMappingEntry;
class ActionNode;
class BeliefNode;

class ActionMapping {
public:
    ActionMapping(BeliefNode *owner) :
        owner_(owner) {
    }
    virtual ~ActionMapping() = default;
    _NO_COPY_OR_MOVE(ActionMapping);

    /* -------------- Association with a belief node ---------------- */
    /** Returns the belief node that owns this mapping. */
    BeliefNode *getOwner() const {
        return owner_;
    }

    /* -------------- Creation and retrieval of nodes. ---------------- */
    /** Retrieves the action node (if any) corresponding to this action. */
    virtual ActionNode *getActionNode(Action const &action) const = 0;
    /** Creates a new action node for the given action. */
    virtual ActionNode *createActionNode(Action const &action) = 0;
    /** Returns the number of child nodes associated with this mapping. */
    virtual long getNChildren() const = 0;

    /* -------------- Retrieval of mapping entries. ---------------- */
    /** Returns the number of entries in this mapping with a nonzero visit
     * count (some of these may not have an associated action node, so this
     * is different to the number of child nodes).
     */
    virtual long getNumberOfVisitedEntries() const = 0;
    /** Returns all of the visited entries in this mapping - some may have
     * null action nodes if the visit counts were initialized to nonzero
     * values.
     */
    virtual std::vector<ActionMappingEntry const *> getVisitedEntries() const = 0;

    /** Returns the mapping entry (if any) associated with the given action. */
    virtual ActionMappingEntry *getEntry(Action const &action) = 0;
    /** Returns the mapping entry (if any) associated with the given action. */
    virtual ActionMappingEntry const *getEntry(Action const &action) const = 0;

    /* ------------------ Methods for unvisited actions ------------------- */
    /** Returns the next action to be tried for this node, or nullptr if there are no more. */
    virtual std::unique_ptr<Action> getNextActionToTry() = 0;

    /* -------------- Retrieval of general statistics. ---------------- */
    /** Returns the total number of times children have been visited. */
    virtual long getTotalVisitCount() const = 0;

    /* --------------- Methods for updating the values ----------------- */
    /** Updates the given action, by adding the given number of visits and the
     * given change in the total q-value.
     *
     * Returns true if and only if the q value of the action changed.
     */
    virtual bool update(Action const &action, long deltaNVisits, double deltaTotalQ) {
        return getEntry(action)->update(deltaNVisits, deltaTotalQ);
    }
private:
    BeliefNode *owner_;
};

} /* namespace solver */

#endif /* SOLVER_ACTIONMAPPING_HPP_ */