/** @file TagState.cpp
 *
 * Contains the implementation for the methods of TagState.
 */
#include "TagState.hpp"

#include <cstddef>                      // for size_t

#include <functional>   // for hash
#include <ostream>                      // for operator<<, ostream, basic_ostream>
#include <vector>
#include <string>
#include "global.hpp"
#include "problems/shared/GridPosition.hpp"  // for GridPosition, operator==, operator<<
#include "solver/abstract-problem/State.hpp"             // for State

namespace tag {
TagState::TagState(GridPosition robotPos, GridPosition opponentPos,
        bool _isTagged, int _timestep) :
    solver::Vector(),
    robotPos_(robotPos),
    opponentPos_(opponentPos),
    isTagged_(_isTagged),
    timestep_(_timestep)
{
}

TagState::TagState(TagState const &other) :
        TagState(other.robotPos_, other.opponentPos_, other.isTagged_,other.timestep_) {
}

std::unique_ptr<solver::Point> TagState::copy() const {
    return std::make_unique<TagState>(robotPos_, opponentPos_, isTagged_,timestep_);
}

double TagState::distanceTo(solver::State const &otherState) const {
    TagState const &otherTagState = static_cast<TagState const &>(otherState);
    double distance = robotPos_.manhattanDistanceTo(otherTagState.robotPos_);
    distance += opponentPos_.manhattanDistanceTo(otherTagState.opponentPos_);
    distance += (isTagged_ == otherTagState.isTagged_) ? 0 : 1;
    distance += abs(timestep_ - otherTagState.timestep_);
    return distance;
}

bool TagState::equals(solver::State const &otherState) const {
    TagState const &otherTagState = static_cast<TagState const &>(otherState);
    return (robotPos_ == otherTagState.robotPos_
            && opponentPos_ == otherTagState.opponentPos_
            && isTagged_ == otherTagState.isTagged_
            && timestep_ == otherTagState.timestep_);
}

std::size_t TagState::hash() const {
    std::size_t hashValue = 0;
    tapir::hash_combine(hashValue, robotPos_.i);
    tapir::hash_combine(hashValue, robotPos_.j);
    tapir::hash_combine(hashValue, opponentPos_.i);
    tapir::hash_combine(hashValue, opponentPos_.j);
    tapir::hash_combine(hashValue, timestep_);
    tapir::hash_combine(hashValue, isTagged_);
    return hashValue;
}

std::vector<double> TagState::asVector() const {
    std::vector<double> vec(6);
    vec[0] = robotPos_.i;
    vec[1] = robotPos_.j;
    vec[2] = opponentPos_.i;
    vec[3] = opponentPos_.j;
    vec[4] = isTagged_ ? 1 : 0;
    vec[6] = timestep_;
    return vec;
}

void TagState::print(std::ostream &os) const {
    os << "ROBOT: " << robotPos_ << " OPPONENT: " << opponentPos_ << " TIMESTEP: " << timestep_;
    if (isTagged_) {
        os << " TAGGED!";
    }
}


GridPosition TagState::getRobotPosition() const {
    return robotPos_;
}

GridPosition TagState::getOpponentPosition() const {
    return opponentPos_;
}

int TagState::getTimestep() const {
    return timestep_;
}
bool TagState::isTagged() const {
    return isTagged_;
}
} /* namespace tag */
