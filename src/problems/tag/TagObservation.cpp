/** @file TagObservation.cpp
 *
 * Contains the implementations for the methods of TagObservation.
 */
#include "TagObservation.hpp"

#include <cstddef>                      // for size_t

#include <algorithm>                    // for copy
#include <iterator>                     // for ostream_iterator
#include <ostream>                      // for operator<<, ostream
#include <vector>                       // for vector, operator==, _Bit_const_iterator, _Bit_iterator_base, hash, vector<>::const_iterator
#include <cfloat>

#include "global.hpp"

#include "problems/shared/GridPosition.hpp"  // for GridPosition, operator==, operator<<
#include "solver/abstract-problem/Observation.hpp"             // for Observation

#include "TagModel.hpp"

namespace tag {

TagObservation::TagObservation(GridPosition position,long timestep,   double _opponentDistance) :
                    position_(position),
                    timestep_(timestep),
                    opponentDistance_(_opponentDistance){
}
std::unique_ptr<solver::Observation>
TagObservation::copy() const {
    return std::make_unique<TagObservation>(position_, timestep_,opponentDistance_);
}

double TagObservation::distanceTo(
        solver::Observation const &otherObs) const {
    TagObservation const &other =
            static_cast<TagObservation const &>(otherObs);
    auto distance = opponentDistance_ - other.opponentDistance_;
    return distance;
}

bool TagObservation::equals(
        solver::Observation const &otherObs) const {
    TagObservation const &other =
        static_cast<TagObservation const &>(otherObs);
    return position_ == other.position_ &&  fabs(opponentDistance_ - other.opponentDistance_) < FLT_EPSILON;
}

std::size_t TagObservation::hash() const {
    std::size_t hashValue = 0;
    tapir::hash_combine(hashValue, position_.i);
    tapir::hash_combine(hashValue, position_.j);
    tapir::hash_combine(hashValue, opponentDistance_);
    return hashValue;
}

void TagObservation::print(std::ostream &os) const {
    os << position_ << " ";
    if (atOpponentPosition()) {
        os << "SEEN";
    } else {
        os << "UNSEEN";
    }
}

GridPosition TagObservation::getPosition() const {
    return position_;
}

bool TagObservation::atOpponentPosition() const {
    return fabs(opponentDistance_) < FLT_EPSILON;
}

long TagObservation::getTimestep() const {
    return timestep_;
}


}
/* namespace tag */
