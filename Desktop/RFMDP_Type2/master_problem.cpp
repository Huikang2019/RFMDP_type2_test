#include "parameter.h"
#include "function.h"

extern int      neighbors[number_of_computers][number_of_neighbors];
extern double   prob[number_of_states][number_of_states][number_of_states][number_of_states];
extern double   prob_real[number_of_states][number_of_states][number_of_states][number_of_states][number_of_computers];
extern int      basis[number_of_computers][scope];

extern int      state_constr[number_of_computers][number_of_constraints];
extern int      state_half_constr[number_of_half][number_of_constraints];
extern int      action_constr[number_of_computers][number_of_constraints];
extern int      time_constr[number_of_constraints];


double mastered_problem_weight(int constr_num, NumArray3D& weight, NumArray3D& weight_half) {
    GRBEnv* env = new GRBEnv();
    GRBModel model = GRBModel(env);

    NumVar3D alpha;
    NumVar3D beta;
    GRBLinExpr obj_func;
    NumArray3D prob_sa;
    int current_state[number_of_computers], original_state[number_of_computers], current_action[number_of_computers];
    double prob_i, prob_j;
    int i, j;

    for (int i = 0; i < number_of_states; i++) {
        for (int j = 0; j < number_of_states; j++) {
            for (int k = 0; k < number_of_computers; k++) {
                alpha[i][j][k] = model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
                beta[i][j][k] = model.addVar(0, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
                obj_func += alpha[i][j][k] / pow(number_of_states, scope);
            }
        }
    }

    model.setObjective(obj_func, GRB_MINIMIZE);

    std::cout << "master problem start ok" << endl;

    for (int constr = 0; constr < constr_num; constr++) {
        for (int k = 0; k < number_of_computers; k++) {
            current_state[k] = state_constr[k][constr];
            original_state[k] = state_constr[k][constr];
            current_action[k] = action_constr[k][constr];
        }        
        if (time_constr[constr] == 0) {
            GRBLinExpr curr_LHS, curr_RHS;
            for (int k = 0; k < number_of_computers; k += 2)
                curr_LHS -= current_state[k];
            for (int k = 0; k < number_of_computers; k++) {
                curr_LHS += alpha[current_state[basis[k][0]]][current_state[basis[k][1]]][k];
            }
            for (int state_i = 0; state_i < number_of_states; ++state_i) {
                for (int state_j = 0; state_j < number_of_states; state_j++) {
                    for (int k = 0; k < number_of_computers; k++) {
                        i = basis[k][0];
                        prob_i = prob_out(current_state, original_state, i, current_action[i], state_i, 0);
                        j = basis[k][1];
                        prob_j = prob_out(current_state, original_state, j, current_action[j], state_j, 0);
                        curr_RHS += prob_i * prob_j * beta[state_i][state_j][k];
                    }
                }
            }
            model.addConstr(curr_RHS * sqrt(discount_factor) - curr_LHS <= 0);
            curr_LHS.clear(); curr_RHS.clear();
        }
        else {
            for (int k = 0; k < number_of_half; k++) {
                original_state[2 * k] = state_half_constr[k][constr];
            }
            GRBLinExpr curr_LHS, curr_RHS;
            for (int k = 1; k < number_of_computers; k += 2)
                curr_LHS -= current_state[k] / sqrt(discount_factor);
            for (int k = 0; k < number_of_computers; k++) {
                curr_LHS += beta[current_state[basis[k][0]]][current_state[basis[k][1]]][k];
            }
            for (int state_i = 0; state_i < number_of_states; ++state_i) {
                for (int state_j = 0; state_j < number_of_states; state_j++) {
                    for (int k = 0; k < number_of_computers; k++) {
                        i = basis[k][0];
                        prob_i = prob_out(current_state, original_state, i, current_action[i], state_i, 1);
                        j = basis[k][1];
                        prob_j = prob_out(current_state, original_state, j, current_action[j], state_j, 1);
                        curr_RHS += prob_i * prob_j * alpha[state_i][state_j][k];
                    }
                }
            }
            model.addConstr(sqrt(discount_factor) * curr_RHS - curr_LHS <= 0);
            curr_LHS.clear(); curr_RHS.clear();
        }
    }

    model.set(GRB_IntParam_OutputFlag, 0);
    model.set(GRB_IntParam_Threads, number_of_Threads);

    model.optimize();

    cout << "number of constraints:" << model.get(GRB_IntAttr_NumConstrs) << "  ";
    cout << "master problem tackled in : " << model.get(GRB_DoubleAttr_Runtime) << " seconds." << endl;

    for (int i = 0; i < number_of_states; i++) {
        for (int j = 0; j < number_of_states; j++) {
            for (int k = 0; k < number_of_computers; k++) {
                weight[i][j][k] = alpha[i][j][k].get(GRB_DoubleAttr_X);
                weight_half[i][j][k] = beta[i][j][k].get(GRB_DoubleAttr_X);
            }
        }
    }

    double obj_value = model.get(GRB_DoubleAttr_ObjVal);

    delete env;

    return obj_value;
}


void creat_mastered_problem_NonRobust(GRBModel& model, NumVar3D& alpha) {
    GRBEnv env = model.getEnv();
    GRBLinExpr obj_func;

    for (int i = 0; i < number_of_states; i++) {
        for (int j = 0; j < number_of_states; j++) {
            for (int k = 0; k < number_of_computers; k++) {
                alpha[i][j][k] = model.addVar(-bigM, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
                obj_func += alpha[i][j][k] / pow(number_of_states, scope);
            }
        }
    }

    model.setObjective(obj_func, GRB_MINIMIZE);
    model.set(GRB_IntParam_OutputFlag, 0);
    model.set(GRB_IntParam_Threads, number_of_Threads);
    model.update();
}

void add_constraint_to_master_problem(GRBModel& model, NumVar3D& alpha,
    const int* current_state, const int* current_action, int& constr_num) {
    GRBEnv env = model.getEnv();

    GRBLinExpr curr_LHS;
    for (int i = 0; i < number_of_computers; ++i)
        curr_LHS -= current_state[i];
    for (int k = 0; k < number_of_computers; k++) {
        curr_LHS += alpha[current_state[basis[k][0]]][current_state[basis[k][1]]][k];
    }

    GRBLinExpr curr_RHS;
    double prob_i, prob_j;
    int i, j;
    for (int state_i = 0; state_i < number_of_states; ++state_i) {
        for (int state_j = 0; state_j < number_of_states; state_j++) {
            for (int k = 0; k < number_of_computers; k++) {
                i = basis[k][0];
                if (current_action[i] == 1) {
                    if (state_i == number_of_states - 1) prob_i = 1.0;
                    else prob_i = 0.0;
                }
                else {
                    prob_i = prob[state_i][current_state[i]][current_state[neighbors[i][0]]][current_state[neighbors[i][1]]];
                }
                j = basis[k][1];
                if (current_action[j] == 1) {
                    if (state_j == number_of_states - 1) prob_j = 1.0;
                    else prob_j = 0.0;
                }
                else {
                    prob_j = prob[state_j][current_state[j]][current_state[neighbors[j][0]]][current_state[neighbors[j][1]]];
                }
                curr_RHS += prob_i * prob_j * alpha[state_i][state_j][k];
            }
        }
    }
    model.addConstr(discount_factor * curr_RHS - curr_LHS <= 0);
    curr_LHS.clear(); curr_RHS.clear();

    //env.end();
}


void add_constraint_to_master_problem_true(GRBModel& model, NumVar3D& alpha,
    const int* current_state, const int* current_action) {
    GRBEnv env = model.getEnv();

    GRBLinExpr curr_LHS;
    for (int i = 0; i < number_of_computers; ++i)
        curr_LHS -= current_state[i];
    for (int k = 0; k < number_of_computers; k++) {
        curr_LHS += alpha[current_state[basis[k][0]]][current_state[basis[k][1]]][k];
    }

    GRBLinExpr curr_RHS;
    double prob_i, prob_j;
    int i, j;
    for (int state_i = 0; state_i < number_of_states; ++state_i) {
        for (int state_j = 0; state_j < number_of_states; state_j++) {
            for (int k = 0; k < number_of_computers; k++) {
                i = basis[k][0];
                if (current_action[i] == 1) {
                    if (state_i == number_of_states - 1) prob_i = 1.0;
                    else prob_i = 0.0;
                }
                else {
                    prob_i = prob_real[state_i][current_state[i]][current_state[neighbors[i][0]]][current_state[neighbors[i][1]]][i];
                }
                j = basis[k][1];
                if (current_action[j] == 1) {
                    if (state_j == number_of_states - 1) prob_j = 1.0;
                    else prob_j = 0.0;
                }
                else {
                    prob_j = prob_real[state_j][current_state[j]][current_state[neighbors[j][0]]][current_state[neighbors[j][1]]][j];
                }
                curr_RHS += prob_i * prob_j * alpha[state_i][state_j][k];
            }
        }
    }
    model.addConstr(discount_factor * curr_RHS - curr_LHS <= 0);
    curr_LHS.clear(); curr_RHS.clear();

    //env.end();
}