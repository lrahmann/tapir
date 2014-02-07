#ifndef SOLVER_PARTICLESET_HPP_
#define SOLVER_PARTICLESET_HPP_

#include <unordered_map>
#include <vector>

#include "defs.hpp"

namespace solver {
class HistoryEntry;

class ParticleSet {
  public:
    ParticleSet();
    ~ParticleSet() = default;

    std::vector<HistoryEntry *>::const_iterator begin() const;
    std::vector<HistoryEntry *>::const_iterator end() const;
    unsigned long size() const;

    void add(HistoryEntry *entry);
    void remove(HistoryEntry *entry);
    HistoryEntry *get(unsigned long index) const;
    bool contains(HistoryEntry *entry) const;

  private:
    std::unordered_map<HistoryEntry *, unsigned long> particleMap_;
    std::vector<HistoryEntry *> particles_;
};
} /* namespace solver */

#endif /* SOLVER_PARTICLESET_HPP_ */
