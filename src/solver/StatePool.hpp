#ifndef SOLVER_STATEPOOL_HPP_
#define SOLVER_STATEPOOL_HPP_

#include <cstddef>                      // for size_t

#include <map>                          // for multimap
#include <memory>                       // for unique_ptr
#include <unordered_map>                // for unordered_map
#include <unordered_set>                // for unordered_set
#include <vector>                       // for vector

#include "ChangeFlags.hpp"               // for ChangeFlags
#include "State.hpp"                    // for State, operator==
#include "StateInfo.hpp"                // for StateInfo

namespace solver {
class StateIndex;

class StatePool {
    friend class Solver;
  public:
    struct Hash {
        std::size_t operator()(State const *state) const {
            return state->hash();
        }
    };
    struct EqualityTest {
        bool operator()(State const *s1,
                State const *s2) const {
            return *s1 == *s2;
        }
    };
    typedef std::unordered_map<State const *, StateInfo *,
            Hash, EqualityTest> StateInfoMap;

    friend class TextSerializer;

    StatePool(long nSDim);
    ~StatePool();
    StatePool(StatePool const &) = delete;
    StatePool(StatePool &&) = delete;
    StatePool &operator=(StatePool const &) = delete;
    StatePool &operator=(StatePool &&) = delete;

    void reset();

    StateInfo *getInfo(State const &state) const;
    StateInfo *getInfoById(long stId) const;

    StateInfo *add(std::unique_ptr<StateInfo> stateInfo);
    StateInfo *createOrGetInfo(State const &state);

    StateIndex *getStateIndex() const;
    void addToStateIndex(StateInfo *stateInfo) const;

    void resetChangeFlags(StateInfo *stateInfo);
    void setChangeFlags(StateInfo *stateInfo, ChangeFlags flags);
    void resetAffectedStates();
    std::unordered_set<StateInfo *> getAffectedStates() const;

  private:
    long nSDim_;
    StateInfoMap stateInfoMap_;
    std::vector<std::unique_ptr<StateInfo>> statesByIndex_;
    std::unique_ptr<StateIndex> stateIndex_;

    std::unordered_set<StateInfo *> changedStates_;
};
} /* namespace solver */

#endif /* SOLVER_STATEPOOL_HPP_ */