/** @file TagModel.cpp
 *
 * Contains the implementations for the core functionality of the Tag POMDP.
 */
#include "TagModel.hpp"

#include <cmath>                        // for floor, pow
#include <cstddef>                      // for size_t
#include <cstdlib>                      // for exit

#include <memory>
#include <fstream>                      // for ifstream, basic_istream, basic_istream<>::__istream_type
#include <iomanip>                      // for operator<<, setw
#include <iostream>                     // for cout
#include <random>                       // for uniform_int_distribution, bernoulli_distribution
#include <unordered_map>                // for _Node_iterator, operator!=, unordered_map<>::iterator, _Node_iterator_base, unordered_map
#include <utility>                      // for make_pair, move, pair
#include <typeinfo>
#include <sstream>

#include "global.hpp"                     // for RandomGenerator, make_unique
#include "problems/shared/GridPosition.hpp"  // for GridPosition, operator==, operator!=, operator<<
#include "problems/shared/ModelWithProgramOptions.hpp"  // for ModelWithProgramOptions

#include "solver/abstract-problem/Action.hpp"            // for Action
#include "solver/abstract-problem/Model.hpp"             // for Model::StepResult, Model
#include "solver/abstract-problem/Observation.hpp"       // for Observation
#include "solver/abstract-problem/State.hpp"             // for State, operator<<, operator==

#include "solver/changes/ChangeFlags.hpp"        // for ChangeFlags

#include "solver/indexing/FlaggingVisitor.hpp"
#include "solver/indexing/RTree.hpp"
#include "solver/indexing/SpatialIndexVisitor.hpp"             // for State, operator<<, operator==

#include "solver/mappings/actions/enumerated_actions.hpp"
#include "solver/mappings/observations/discrete_observations.hpp"

#include "solver/ActionNode.hpp"
#include "solver/BeliefNode.hpp"
#include "solver/StatePool.hpp"

#include "TagAction.hpp"
#include "TagObservation.hpp"
#include "TagOptions.hpp"
#include "TagState.hpp"                 // for TagState
#include "TagTextSerializer.hpp"

using std::cout;
using std::endl;

namespace tag {
TagUBParser::TagUBParser(TagModel *model) :
        model_(model) {
}
solver::HeuristicFunction TagUBParser::parse(solver::Solver */*solver*/, std::vector<std::string> /*args*/) {
    return [this] (solver::HistoryEntry const *, solver::State const *state,
            solver::HistoricalData const *) {
        return model_->getUpperBoundHeuristicValue(*state);
    };
}

TagModel::TagModel(RandomGenerator *randGen, std::unique_ptr<TagOptions> options) :
            ModelWithProgramOptions("Tag", randGen, std::move(options)),
            options_(const_cast<TagOptions *>(static_cast<TagOptions const *>(getOptions()))),
            moveCost_(options_->moveCost),
            tagReward_(options_->tagReward),
            failedTagPenalty_(options_->failedTagPenalty),
            opponentStayProbability_(options_->opponentStayProbability),
            nRows_(0), // to be updated
            nCols_(0), // to be updated
            mapText_(), // will be pushed to
            dynamic_obj(),
            envMap_(), // will be pushed to
            nActions_(5),
            maxTime_(100),
            mdpSolver_(nullptr),
            pairwiseDistances_()
            {
    options_->numberOfStateVariables = 5;
    options_->minVal = -failedTagPenalty_ / (1 - options_->discountFactor);
    options_->maxVal = tagReward_;

    // Register the upper bound heuristic parser.
    registerHeuristicParser("upper", std::make_unique<TagUBParser>(this));
    // Register the exact MDP heuristic parser.
    registerHeuristicParser("exactMdp", std::make_unique<TagMdpParser>(this));

    // Read the map from the file.
    std::ifstream inFile;
    inFile.open(options_->mapPath);
    if (!inFile.is_open()) {
        std::ostringstream message;
        message << "ERROR: Failed to open " << options_->mapPath;
        debug::show_message(message.str());
        std::exit(1);
    }
    inFile >> nRows_ >> nCols_;
    std::string tmp;
    getline(inFile, tmp);
    for (long i = 0; i < nRows_; i++) {
        getline(inFile, tmp);
        mapText_.push_back(tmp);
    }


    inFile >> maxTime_;
    getline(inFile, tmp);
    for (long i = 0; i < maxTime_; i++) {
        getline(inFile, tmp);
        dynamic_obj.push_back(tmp);
    }

    // cout << "dynamic object" << dynamic_obj << endl;

    inFile.close();

    initialize();
    if (options_->hasVerboseOutput) {
        cout << "Constructed the TagModel" << endl;
        cout << "Discount: " << options_->discountFactor << endl;
        cout << "Size: " << nRows_ << " by " << nCols_ << endl;
        cout << "move cost: " << moveCost_ << endl;
        cout << "maximumtime: " << maxTime_ << endl;
        cout << "nActions: " << nActions_ << endl;
        cout << "nStVars: " << options_->numberOfStateVariables << endl;
        cout << "minParticleCount: " << options_->minParticleCount << endl;
        cout << "Environment:" << endl << endl;
        drawEnv(cout);
    }
}

int TagModel::getMapDistance(GridPosition p1, GridPosition p2,long t1 = 0 , long t2 = 0) {
    return pairwiseDistances_[p1.i][p1.j][p2.i][p2.j];
}

void TagModel::calculateDistancesFrom(GridPosition position,long timestep = 0) {

    auto &distanceGrid = pairwiseDistances_[position.i][position.j];

    // Fill the grid with "-1", for inaccessible cells.
    for (auto &row : distanceGrid) {
        for (auto &cell : row) {
                cell = -1;

        }
    }
    if (envMap_[0][position.i][position.j] == TagCellType::WALL) {
        return;
    }

    std::queue<GridPosition> queue;
    // Start at 0 for the current position.
    distanceGrid[position.i][position.j] = 0;
    queue.push(position);
    while (!queue.empty()) {
        GridPosition pos;
        pos = queue.front();
        queue.pop();
        int distance = distanceGrid[pos.i][pos.j] + 1;
        for (ActionType direction : { ActionType::NORTH, ActionType::SOUTH, ActionType::WEST,
                ActionType::EAST }) {
            GridPosition nextPos;
            bool isLegal;
            std::tie(nextPos, isLegal) = getMovedPos(pos,0, direction);
            // If it's legal and it's an improvement it needs to be queued.
            if (isLegal) {
                int &nextPosDistance = distanceGrid[nextPos.i][nextPos.j];
                if (nextPosDistance == -1 || nextPosDistance > distance) {
                    nextPosDistance = distance;
                    queue.push(nextPos);
                }
            }
        }
    }
}

void TagModel::calculatePairwiseDistances() {
    for (long timeStep=0;timeStep<maxTime_;timeStep++) {

        for (int i = 0; i < nRows_; i++) {
            for (int j = 0; j < nCols_; j++) {
                calculateDistancesFrom(GridPosition(i, j),timeStep);
            }
        }
    }
}

void TagModel::initialize() {
    GridPosition p;
    //build massive array


    envMap_ = make_vector<TagCellType >(maxTime_,nRows_,nCols_);

    for (p.i = nRows_ - 1; p.i >= 0; p.i--) {
        for (p.j = 0; p.j < nCols_; p.j++) {
            char c = mapText_[p.i][p.j];
            TagCellType cellType;
            if (c == 'X') {
                cellType = TagCellType::WALL;
            } else {
                cellType = TagCellType::EMPTY;
            }


            for(int t=0;t<maxTime_;t++){
                envMap_[t][p.i][p.j] = cellType;
            }
        }
    }

    int maxTime = -1;
    int lastX;
    int lastY;
    for (auto line : dynamic_obj){
        std::stringstream stream(line);
        int x ,y,time;
        stream >> x >> y >> time;
        //cout << x << " " << y << " " << time << endl;
        envMap_[time][x][y] = TagCellType:: WALL;
        if(time > maxTime){
            lastX =x;
            lastY =y;
            maxTime=time;

        }
    }

    if(maxTime>=0) {
        for (int i = maxTime + 1; i < maxTime_; i++) {
            envMap_[i][lastX][lastY] = TagCellType::WALL;
        }
    }


    pairwiseDistances_ = make_vector<int>(nRows_,nCols_,nRows_,nCols_);

    calculatePairwiseDistances();
}

std::tuple<GridPosition,int> TagModel::randomEmptyCell(long timeStep) {
    GridPosition pos;
    while (true) {
        pos.i = std::uniform_int_distribution<long>(0, nRows_ - 1)(
                *getRandomGenerator());
        pos.j = std::uniform_int_distribution<long>(0, nCols_ - 1)(
                *getRandomGenerator());
        if (timeStep == -1) {
            timeStep = std::uniform_int_distribution<long>(0, maxTime_ - 1)(*getRandomGenerator());
        }
        if (envMap_[timeStep][pos.i][pos.j] == TagCellType::EMPTY) {
            break;
        }
    }
    return std::tie(pos,timeStep);
}


/* --------------- The model interface proper ----------------- */
std::unique_ptr<solver::State> TagModel::sampleAnInitState() {
    GridPosition robotPos;
    long timestep = 0;
    std::tie(robotPos,timestep)= randomEmptyCell(timestep);
    GridPosition opponentPos = std::get<0>(randomEmptyCell(timestep));
    return std::make_unique<TagState>(robotPos, opponentPos, false, timestep);
}

std::unique_ptr<solver::State> TagModel::sampleStateUninformed() {
    GridPosition robotPos;
    long timestep = -1;
    std::tie(robotPos,timestep)= randomEmptyCell(timestep);
    GridPosition opponentPos = std::get<0>(randomEmptyCell(timestep));
    return std::make_unique<TagState>(robotPos, opponentPos, false, timestep);
}

bool TagModel::isTerminal(solver::State const &state) {
    auto tagState = static_cast<TagState const &>(state);
    return tagState.isTagged() || tagState.getTimestep()>=maxTime_-1;
}

bool TagModel::isValid(solver::State const &state) {
    TagState const tagState = static_cast<TagState const &>(state);
    return isValid(tagState.getOpponentPosition(),tagState.getTimestep()) && isValid(tagState.getRobotPosition(),tagState.getTimestep());
}


/* -------------------- Black box dynamics ---------------------- */
std::pair<std::unique_ptr<TagState>, bool> TagModel::makeNextState(
        solver::State const &state, solver::Action const &action) {
    TagState const &tagState = static_cast<TagState const &>(state);
    if (tagState.isTagged()) {
        return std::make_pair(std::make_unique<TagState>(tagState), false);
    }
    TagAction const &tagAction = static_cast<TagAction const &>(action);

    long nexTimeStep = std::min((long)tagState.getTimestep()+1,maxTime_);

    GridPosition robotPos = tagState.getRobotPosition();
    GridPosition opponentPos = tagState.getOpponentPosition();
    if (tagAction.getActionType() == ActionType::TAG
            && robotPos == opponentPos) {
        return std::make_pair(
                std::make_unique<TagState>(robotPos, opponentPos, true,nexTimeStep), true);
    }

    GridPosition newOpponentPos = sampleNextOpponentPosition(robotPos, opponentPos,tagState.getTimestep());
    GridPosition newRobotPos;
    bool wasValid;
    std::tie(newRobotPos, wasValid) = getMovedPos(robotPos, tagState.getTimestep(),tagAction.getActionType());
    return std::make_pair(std::make_unique<TagState>(newRobotPos, newOpponentPos, false,nexTimeStep),
            wasValid);
}

std::vector<ActionType> TagModel::makeOpponentActions(
        GridPosition const &robotPos, GridPosition const &opponentPos) {
    std::vector<ActionType> actions;
    if (robotPos.i > opponentPos.i) {
        actions.push_back(ActionType::NORTH);
        actions.push_back(ActionType::NORTH);
    } else if (robotPos.i < opponentPos.i) {
        actions.push_back(ActionType::SOUTH);
        actions.push_back(ActionType::SOUTH);
    } else {
        actions.push_back(ActionType::NORTH);
        actions.push_back(ActionType::SOUTH);
    }
    if (robotPos.j > opponentPos.j) {
        actions.push_back(ActionType::WEST);
        actions.push_back(ActionType::WEST);
    } else if (robotPos.j < opponentPos.j) {
        actions.push_back(ActionType::EAST);
        actions.push_back(ActionType::EAST);
    } else {
        actions.push_back(ActionType::EAST);
        actions.push_back(ActionType::WEST);
    }
    return actions;
}

/** Generates a proper distribution for next opponent positions. */
std::unordered_map<GridPosition, double> TagModel::getNextOpponentPositionDistribution(
        GridPosition const &robotPos, GridPosition const &opponentPos, long timeStep) {
    std::vector<ActionType> actions = makeOpponentActions(robotPos, opponentPos);
    std::unordered_map<GridPosition, double> distribution;
    double actionProb = (1 - opponentStayProbability_) / actions.size();
    for (ActionType action : actions) {
        distribution[getMovedPos(opponentPos, timeStep,action).first] += actionProb;
    }
    distribution[opponentPos] += opponentStayProbability_;
    return std::move(distribution);
}

GridPosition TagModel::sampleNextOpponentPosition(GridPosition const &robotPos,
        GridPosition const &opponentPos, long const &timestep) {
    // Randomize to see if the opponent stays still.
    if (std::bernoulli_distribution(opponentStayProbability_)(
            *getRandomGenerator())) {
        return opponentPos;
    }
    std::vector<ActionType> actions(makeOpponentActions(robotPos, opponentPos));
    ActionType action = actions[std::uniform_int_distribution<long>(0,
            actions.size() - 1)(*getRandomGenerator())];
    return getMovedPos(opponentPos, timestep, action).first;
}

std::pair<GridPosition, bool> TagModel::getMovedPos(GridPosition const &position,long timestep,
        ActionType action) {
    GridPosition movedPos = position;
    switch (action) {
    case ActionType::NORTH:
        movedPos.i -= 1;
        break;
    case ActionType::EAST:
        movedPos.j += 1;
        break;
    case ActionType::SOUTH:
        movedPos.i += 1;
        break;
    case ActionType::WEST:
        movedPos.j -= 1;
        break;
    case ActionType::TAG:
        break;
    default:
        std::ostringstream message;
        message << "Invalid action: " << (long) action;
        debug::show_message(message.str());
        break;
    }
    bool wasValid = isValid(movedPos,timestep+1);
    if (!wasValid) {
        movedPos = position;
    }
    return std::make_pair(movedPos, wasValid);
}

bool TagModel::isValid(GridPosition const &position, long const &timeStep) {
    if(timeStep >= maxTime_)
        return isValid(position,maxTime_-1);
    return (position.i >= 0 && position.i < nRows_ && position.j >= 0
            && position.j < nCols_ && timeStep>= 0 && timeStep < maxTime_ && envMap_[timeStep][position.i][position.j] != TagCellType::WALL);
}

double TagModel::distanceToOpponent(TagState const &state){
    double stepI =  state.getOpponentPosition().i - state.getRobotPosition().i ;
    double stepJ =  state.getOpponentPosition().j - state.getRobotPosition().j ;
    double norm = sqrt(stepI*stepI + stepJ*stepJ);
    if(norm == 0){
        return true;
    }
    if(norm > 3){
        return -1;
    }
    stepJ /= norm;
    stepI/= norm;
    for(int i=1;i<norm;i++){
        int x = (int) state.getRobotPosition().i + stepI*i;
        int y = (int) state.getRobotPosition().j + stepJ*i;

        if(envMap_[state.getTimestep()][x][y] == TagCellType::WALL)
            return -1;
    }
    return norm;
}


std::unique_ptr<solver::Observation> TagModel::makeObservation(TagState const &nextState) {
    return std::make_unique<TagObservation>(nextState.getRobotPosition(),nextState.getTimestep(),
                                            this->distanceToOpponent(nextState)
    );
}

double TagModel::generateReward(solver::State const &state,
        solver::Action const &action,
        solver::TransitionParameters const */*tp*/,
        solver::State const *nextState) {
    TagState const &tagState = static_cast<TagState const &>(state);
    TagState const &nextTagState = static_cast<TagState const &>(*nextState);
    if (static_cast<TagAction const &>(action).getActionType()
            == ActionType::TAG) {
        if (tagState.getRobotPosition() == tagState.getOpponentPosition()) {
            return tagReward_;
        } else {
            return -failedTagPenalty_;
        }
    } else {
        auto nextDistance = nextTagState.getOpponentPosition().manhattanDistanceTo(nextTagState.getRobotPosition());
        auto currentDistance = tagState.getOpponentPosition().manhattanDistanceTo(tagState.getRobotPosition());

        if(nextDistance<currentDistance){
            return -moveCost_/2;
        }

        return -moveCost_;
    }
}

std::unique_ptr<solver::State> TagModel::generateNextState(
        solver::State const &state, solver::Action const &action,
        solver::TransitionParameters const */*tp*/) {
    return makeNextState(static_cast<TagState const &>(state), action).first;
}

std::unique_ptr<solver::Observation> TagModel::generateObservation(
        solver::State const */*state*/, solver::Action const &/*action*/,
        solver::TransitionParameters const */*tp*/,
        solver::State const &nextState) {
    return makeObservation(static_cast<TagState const &>(nextState));
}

solver::Model::StepResult TagModel::generateStep(solver::State const &state,
        solver::Action const &action) {
    solver::Model::StepResult result;
    result.action = action.copy();
    std::unique_ptr<TagState> nextState = makeNextState(state, action).first;

    result.observation = makeObservation(*nextState);
    result.reward = generateReward(state, action, nullptr, nextState.get());
    result.isTerminal = isTerminal(*nextState);
    result.nextState = std::move(nextState);
    return result;
}


/* -------------- Methods for handling model changes ---------------- */
void TagModel::applyChanges(std::vector<std::unique_ptr<solver::ModelChange>> const &changes,
        solver::Solver *solver) {
    solver::StatePool *pool = nullptr;
    if (solver != nullptr) {
        pool = solver->getStatePool();
    }

    solver::HeuristicFunction heuristic = getHeuristicFunction();
    std::vector<double> allHeuristicValues;
    if (pool != nullptr) {
        long nStates = pool->getNumberOfStates();
        allHeuristicValues.resize(nStates);
        for (long index = 0; index < nStates; index++) {
            allHeuristicValues[index] = heuristic(nullptr, pool->getInfoById(index)->getState(),
                    nullptr);
        }
    }

    for (auto const &change : changes) {
        TagChange const &tagChange = static_cast<TagChange const &>(*change);
        if (options_->hasVerboseOutput) {
            cout << tagChange.changeType << " " << tagChange.i0 << " "
                    << tagChange.j0;
            cout << " " << tagChange.i1 << " " << tagChange.j1 << endl;
        }

        TagCellType newCellType;
        if (tagChange.changeType == "Add Obstacles") {
            newCellType = TagCellType::WALL;
        } else if (tagChange.changeType == "Remove Obstacles") {
            newCellType = TagCellType::EMPTY;
        } else {
            cout << "Invalid change type: " << tagChange.changeType;
            continue;
        }

        for (long i = tagChange.i0; i <= tagChange.i1; i++) {
            for (long j = tagChange.j0; j <= tagChange.j1; j++) {
                for( long t = 0 ; t < maxTime_ ; t++) {
                    envMap_[t][i][j] = newCellType;
                }
            }
        }

        if (pool == nullptr) {
            continue;
        }

        solver::RTree *tree = static_cast<solver::RTree *>(pool->getStateIndex());
        if (tree == nullptr) {
            debug::show_message("ERROR: state index must be enabled to handle changes in Tag!");
            std::exit(4);
        }

        double iLo = tagChange.i0;
        double iHi = tagChange.i1;
        double iMx = nRows_ - 1.0;

        double jLo = tagChange.j0;
        double jHi = tagChange.j1;
        double jMx = nCols_ - 1.0;

        // Adding walls => any states where the robot or the opponent are in a wall must
        // be deleted.
        if (newCellType == TagCellType::WALL) {
            solver::FlaggingVisitor visitor(pool, solver::ChangeFlags::DELETED);
            // Robot is in a wall.
            tree->boxQuery(visitor,
                    {iLo, jLo, 0.0, 0.0, 0.0},
                    {iHi, jHi, iMx, jMx, 1.0});
            // Opponent is in a wall.
            tree->boxQuery(visitor,
                    {0.0, 0.0, iLo, jLo, 0.0},
                    {iMx, jMx, iHi, jHi, 1.0});

        }

        // Also, state transitions around the edges of the new / former obstacle must be revised.
        solver::FlaggingVisitor visitor(pool, solver::ChangeFlags::TRANSITION);
        tree->boxQuery(visitor,
                {iLo - 1, jLo - 1, 0.0, 0.0, 0.0},
                {iHi + 1, jHi + 1, iMx, jMx, 1.0});
        tree->boxQuery(visitor,
                {0.0, 0.0, iLo - 1, jLo - 1, 0.0},
                {iMx, jMx, iHi + 1, jHi + 1, 1.0});
    }

    if (mdpSolver_ != nullptr) {
        mdpSolver_->solve();
    }

    calculatePairwiseDistances();

    // Check for heuristic changes.
    if (pool != nullptr) {
        long nStates = pool->getNumberOfStates();
        for (long index = 0; index < nStates; index++) {
            double oldValue = allHeuristicValues[index];
            solver::StateInfo *info = pool->getInfoById(index);
            double newValue = heuristic(nullptr, info->getState(), nullptr);
            if (std::abs(newValue - oldValue) > 1e-5) {
                pool->setChangeFlags(info, solver::ChangeFlags::HEURISTIC);
            }
        }
    }
}


/* ------------ Methods for handling particle depletion -------------- */
std::vector<std::unique_ptr<solver::State>> TagModel::generateParticles(
        solver::BeliefNode */*previousBelief*/, solver::Action const &action,
        solver::Observation const &obs,
        long nParticles,
        std::vector<solver::State const *> const &previousParticles) {
    std::vector<std::unique_ptr<solver::State>> newParticles;
    TagObservation const &observation =
            (static_cast<TagObservation const &>(obs));
    ActionType actionType =
            (static_cast<TagAction const &>(action).getActionType());

    typedef std::unordered_map<TagState, double> WeightMap;
    WeightMap weights;
    double weightTotal = 0;

    GridPosition newRobotPos(observation.getPosition());
    if (observation.atOpponentPosition()) {
        long timestep = static_cast<TagState const *>(previousParticles.front())->getTimestep();
        // If we saw the opponent, we must be in the same place.
        newParticles.push_back(
                std::make_unique<TagState>(newRobotPos, newRobotPos,
                        actionType == ActionType::TAG,timestep));
    } else {
        // We didn't see the opponent, so we must be in different places.
        for (solver::State const *state : previousParticles) {
            TagState const *tagState = static_cast<TagState const *>(state);
            GridPosition oldRobotPos(tagState->getRobotPosition());
            // Ignore states that do not match knowledge of the robot's position.
            if (newRobotPos != getMovedPos(oldRobotPos, tagState->getTimestep(),actionType).first) {
                continue;
            }

            // Get the probability distribution for opponent moves.
            GridPosition oldOpponentPos(tagState->getOpponentPosition());
            std::unordered_map<GridPosition, double> opponentPosDistribution = (
                    getNextOpponentPositionDistribution(oldRobotPos, oldOpponentPos,tagState->getTimestep()));

            for (auto const &entry : opponentPosDistribution) {
                if (entry.first != newRobotPos) {
                    TagState newState(newRobotPos, entry.first, false, tagState->getTimestep()+1);
                    weights[newState] += entry.second;
                    weightTotal += entry.second;
                }
            }
        }
        double scale = nParticles / weightTotal;
        for (WeightMap::iterator it = weights.begin(); it != weights.end();
                it++) {
            double proportion = it->second * scale;
            long numToAdd = static_cast<long>(proportion);
            if (std::bernoulli_distribution(proportion - numToAdd)(
                    *getRandomGenerator())) {
                numToAdd += 1;
            }
            for (int i = 0; i < numToAdd; i++) {
                newParticles.push_back(std::make_unique<TagState>(it->first));
            }
        }
    }
    return newParticles;
}

    std::vector<std::unique_ptr<solver::State>> TagModel::generateParticles(
        solver::BeliefNode *previousBelief, solver::Action const &action,
        solver::Observation const &obs, long nParticles) {
    std::vector<std::unique_ptr<solver::State>> newParticles;
    TagObservation const &observation =
            (static_cast<TagObservation const &>(obs));
    ActionType actionType =
            (static_cast<TagAction const &>(action).getActionType());
    GridPosition newRobotPos(observation.getPosition());
    if (observation.atOpponentPosition()) {
        // If we saw the opponent, we must be in the same place.
        while ((long)newParticles.size() < nParticles) {
            newParticles.push_back(
                std::make_unique<TagState>(newRobotPos, newRobotPos,
                        actionType == ActionType::TAG,this->getCurrentTimestep(previousBelief)));
        }
    } else {
        while ((long)newParticles.size() < nParticles) {
            std::unique_ptr<solver::State> state = sampleStateUninformed();
            solver::Model::StepResult result = generateStep(*state, action);
            if (obs == *result.observation) {
                newParticles.push_back(std::move(result.nextState));
            }
        }
    }
    return newParticles;
}


/* --------------- Pretty printing methods ----------------- */
void TagModel::dispCell(TagCellType cellType, std::ostream &os) {
    switch (cellType) {
    case TagCellType::EMPTY:
        os << " 0";
        break;
    case TagCellType::WALL:
        os << "XX";
        break;
    default:
        os << "ER";
        break;
    }
}

void TagModel::drawEnv(std::ostream &os) {
    drawEnv(os,0);
}
void TagModel::drawEnv(std::ostream &os,long timeStep) {
        for (auto &row : envMap_[timeStep]) {
            for (auto cellType : row) {
                dispCell(cellType, os);
                os << " ";
            }
            os << endl;
    }
}

void TagModel::drawSimulationState(solver::BeliefNode const *belief,
        solver::State const &state, std::ostream &os) {


    TagState const &tagState = static_cast<TagState const &>(state);
    long currentTime = tagState.getTimestep();
    os << "Current Time: " << currentTime << std::endl;
    os << "Current Time belief: " << getCurrentTimestep(belief) << std::endl;
    std::vector<solver::State const *> particles = belief->getStates();
    std::vector<std::vector<long>> particleCounts(nRows_,
            std::vector<long>(nCols_));
    for (solver::State const *particle : particles) {
        GridPosition opponentPos =
                static_cast<TagState const &>(*particle).getOpponentPosition();
        particleCounts[opponentPos.i][opponentPos.j] += 1;
    }

    std::vector<int> colors { 196, 161, 126, 91, 56, 21, 26, 31, 36, 41, 46 };
    if (options_->hasColorOutput) {
        os << "Color map: ";
        for (int color : colors) {
            os << "\033[38;5;" << color << "m";
            os << '*';
            os << "\033[0m";
        }
        os << endl;
    }
    for (std::size_t i = 0; i < nRows_; i++) {
        for (std::size_t j = 0; j < nCols_; j++) {
            double proportion = (double) particleCounts[i][j]
                    / particles.size();
            if (options_->hasColorOutput) {
                if (proportion > 0) {
                    int color = colors[proportion * (colors.size() - 1)];
                    os << "\033[38;5;" << color << "m";
                }
            }
            GridPosition pos(i, j);
            bool hasRobot = (pos == tagState.getRobotPosition());
            bool hasOpponent = (pos == tagState.getOpponentPosition());
            if (hasRobot) {
                if (hasOpponent) {
                    os << "#";
                } else {
                    os << "r";
                }
            } else if (hasOpponent) {
                os << "o";
            } else {
                if (envMap_[currentTime][i][j] == TagCellType::WALL) {
                    os << "X";
                } else {
                    os << ".";
                }
            }
            if (options_->hasColorOutput) {
                os << "\033[0m";
            }
        }
        os << endl;
    }
}


/* ---------------------- Basic customizations  ---------------------- */
double TagModel::getDefaultHeuristicValue(solver::HistoryEntry const */*entry*/,
            solver::State const *state, solver::HistoricalData const */*data*/) {
    TagState const &tagState = static_cast<TagState const &>(*state);
    if (tagState.isTagged()) {
        return 0;
    }
    GridPosition robotPos = tagState.getRobotPosition();
    GridPosition opponentPos = tagState.getOpponentPosition();
    long dist = getMapDistance(robotPos, opponentPos);
    double nSteps = dist / opponentStayProbability_;
    double finalDiscount = std::pow(options_->discountFactor, nSteps);
    double qVal = -moveCost_ * (1 - finalDiscount) / (1 - options_->discountFactor);
    qVal += finalDiscount * tagReward_;
    return qVal;
}

double TagModel::getUpperBoundHeuristicValue(solver::State const &state) {
    TagState const &tagState = static_cast<TagState const &>(state);
    if (tagState.isTagged()) {
        return 0;
    }
    GridPosition robotPos = tagState.getRobotPosition();
    GridPosition opponentPos = tagState.getOpponentPosition();
    long dist = getMapDistance(robotPos, opponentPos);
    double finalDiscount = std::pow(options_->discountFactor, dist);
    double qVal = -moveCost_ * (1 - finalDiscount) / (1 - options_->discountFactor);
    qVal += finalDiscount * tagReward_;
    return qVal;
}


/* ------- Customization of more complex solver functionality  --------- */
std::vector<std::unique_ptr<solver::DiscretizedPoint>> TagModel::getAllActionsInOrder() {
    std::vector<std::unique_ptr<solver::DiscretizedPoint>> allActions;
    for (long code = 0; code < nActions_; code++) {
        allActions.push_back(std::make_unique<TagAction>(code));
    }
    return allActions;
}

std::vector<std::vector<float>> TagModel::getBeliefProportions(solver::BeliefNode const *belief) {
    std::vector<solver::State const *> particles = belief->getStates();
    std::vector<std::vector<long>> particleCounts(nRows_,  std::vector<long>(nCols_));
    for (solver::State const *particle : particles) {
        GridPosition targetPos = static_cast<TagState const &>(*particle).getOpponentPosition();
        particleCounts[targetPos.i][targetPos.j] += 1;
    }

    std::vector<std::vector<float>> result;
    for (std::size_t i = 0; i < nRows_; i++) {
        result.push_back(std::vector<float>());
        for (std::size_t j = 0; j < nCols_; j++) {
            result[i].push_back((float) particleCounts[i][j]/particles.size());
        }
    }
    return result;
}

std::unique_ptr<solver::ActionPool> TagModel::createActionPool(solver::Solver */*solver*/) {
    return std::make_unique<solver::EnumeratedActionPool>(this, getAllActionsInOrder());
}
std::unique_ptr<solver::Serializer> TagModel::createSerializer(solver::Solver *solver) {
    return std::make_unique<TagTextSerializer>(solver);
}

long TagModel::getCurrentTimestep(const solver::BeliefNode* belief) {
    return static_cast<const TagState *>(belief->getStates()[0])->getTimestep();
}


} /* namespace tag */
