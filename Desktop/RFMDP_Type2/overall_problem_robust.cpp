#include "parameter.h"
#include "function.h"

extern int          neighbors[number_of_computers][number_of_neighbors];
extern double       prob[number_of_states][number_of_states][number_of_states][number_of_states];
extern int          basis[number_of_computers][scope];
extern NumArray3D   weight;

void solve_overall_problem_scope_2_robust () {
    // set up Gurobi model
    GRBEnv env = GRBEnv();
    GRBModel model = GRBModel(env);

    NumVar3D alpha, beta;
    GRBQuadExpr obj_func = 0.0;
    for (int i = 0; i < number_of_states; i++) {
        for (int j = 0; j < number_of_states; j++) {
            for (int k = 0; k < number_of_computers; k++) {
                alpha[i][j][k] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
                beta[i][j][k] = model.addVar(-GRB_INFINITY, GRB_INFINITY, 0.0, GRB_CONTINUOUS);
                obj_func += alpha[i][j][k] / pow(number_of_states, scope);
            }
        }
    }

    model.setObjective(obj_func, GRB_MINIMIZE);
    model.update();

    // generate the constraints (one for each state)
    int number_of_constraints = 0, sum_action;
    int current_state[number_of_computers], original_state[number_of_computers], current_action[number_of_computers],
        current_state_half[number_of_half];
    for (int i = 0; i < number_of_computers; ++i) {
        current_state[i] = 0;
        current_action[i] = 0;
        if (i % 2 == 0) current_state_half[i / 2] = 0;
    }

    for (;;) {
        for (int k = 0; k < number_of_computers; k++) {
            original_state[k] = current_state[k];
        }
        for (;;) {
            sum_action = 0;
            for (int i = 0; i < number_of_computers; ++i)
                sum_action += current_action[i];
            if (sum_action <= number_of_actions) {
                // create constraint for current state and action
                GRBLinExpr curr_LHS = 0.0;
                for (int i = 0; i < number_of_computers; ++i)
                    curr_LHS -= current_state[i];
                for (int k = 0; k < number_of_computers; k++) {
                    curr_LHS += alpha[current_state[basis[k][0]]][current_state[basis[k][1]]][k];
                }

                GRBLinExpr curr_RHS = 0.0;
                double prob_i, prob_j;
                int i, j;
                for (int state_i = 0; state_i < number_of_states; ++state_i) {
                    for (int state_j = 0; state_j < number_of_states; ++state_j) {
                        for (int k = 0; k < number_of_computers; k++) {
                            i = basis[k][0];
                            prob_i = prob_out(current_state, original_state, i, current_action[i], state_i, 0);
                            j = basis[k][1];
                            prob_j = prob_out(current_state, original_state, j, current_action[j], state_j, 0);
                            curr_RHS += prob_i * prob_j * beta[state_i][state_j][k];
                        }
                    }
                }
                model.addConstr(curr_LHS >= curr_RHS);
                ++number_of_constraints;
            }

            bool success = false;
            for (int act_update = number_of_computers - 1; act_update >= 0; --act_update) {
                ++current_action[act_update];
                if (current_action[act_update] <= 1) {
                    success = true;
                    break;
                }
                else
                    current_action[act_update] = 0;
            }
            if (!success) break;
        }

        // go to next state or terminate
        bool success = false;
        for (int curr_update = number_of_computers - 1; curr_update >= 0; --curr_update) {
            ++current_state[curr_update];
            if (current_state[curr_update] <= number_of_states - 1) {
                success = true;
                break;
            }
            else
                current_state[curr_update] = 0;
        }
        if (!success) break;
    }

    for (int i = 0; i < number_of_computers; ++i) {
        current_state[i] = 0;
        current_action[i] = 0;
        if (i % 2 == 0) current_state_half[i / 2] = 0;
    }

    for (;;) {
        for (int k = 0; k < number_of_computers; k++) {
            original_state[k] = current_state[k];
        }
        for (;;) {
            for (int k = 0; k < number_of_half; k++) {
                original_state[2 * k] = current_state_half[k];
            }
            for (;;) {
                bool right = false;
                for (int k = 0; k < number_of_half; k++) {
                    if ((current_action[2 * k] == 1) && (current_state[2 * k] < 1)) {
                        right = true;
                    }
                }
                if (!right) {
                    sum_action = 0;
                    for (int i = 0; i < number_of_computers; ++i)
                        sum_action += current_action[i];
                    if (sum_action <= number_of_actions) {
                        // create constraint for current state and action
                        GRBLinExpr curr_LHS = 0.0;
                        for (int k = 0; k < number_of_computers; k++) {
                            curr_LHS += beta[current_state[basis[k][0]]][current_state[basis[k][1]]][k];
                        }

                        GRBLinExpr curr_RHS = 0.0;
                        double prob_i, prob_j;
                        int i, j;
                        for (int state_i = 0; state_i < number_of_states; ++state_i) {
                            for (int state_j = 0; state_j < number_of_states; ++state_j) {
                                for (int k = 0; k < number_of_computers; k++) {
                                    i = basis[k][0];
                                    prob_i = prob_out(current_state, original_state, i, current_action[i], state_i, 1);
                                    j = basis[k][1];
                                    prob_j = prob_out(current_state, original_state, j, current_action[j], state_j, 1);
                                    curr_RHS += prob_i * prob_j * alpha[state_i][state_j][k];
                                }
                            }
                        }
                        model.addConstr(curr_LHS >= discount_factor * curr_RHS);
                        ++number_of_constraints;
                    }
                }


                bool success = false;
                for (int act_update = number_of_computers - 1; act_update >= 0; --act_update) {
                    ++current_action[act_update];
                    if (current_action[act_update] <= 1) {
                        success = true;
                        break;
                    }
                    else
                        current_action[act_update] = 0;
                }
                if (!success) break;
            }
            bool success = false;
            for (int curr_update = number_of_half - 1; curr_update >= 0; --curr_update) {
                ++current_state_half[curr_update];
                if (current_state_half[curr_update] <= number_of_states - 1) {
                    success = true;
                    break;
                }
                else
                    current_state_half[curr_update] = 0;
            }
            if (!success) break;
        }
        // go to next state or terminate
        bool success = false;
        for (int curr_update = number_of_computers - 1; curr_update >= 0; --curr_update) {
            ++current_state[curr_update];
            if (current_state[curr_update] <= number_of_states - 1) {
                success = true;
                break;
            }
            else
                current_state[curr_update] = 0;
        }
        if (!success) break;
    }

    

        
            

    cout << number_of_constraints << " constraints generated." << endl;

    // solve the problem
    model.optimize();

    cout << "objective value of order " << 2 << " is: " << model.get(GRB_DoubleAttr_ObjVal) << endl;
    cout << "problem tackled in " << model.get(GRB_DoubleAttr_Runtime) << " seconds." << endl;

    for (int i = 0; i < number_of_states; i++) {
        for (int j = 0; j < number_of_states; j++) {
            for (int k = 0; k < number_of_computers; k++) {
                weight[i][j][k] = alpha[i][j][k].get(GRB_DoubleAttr_X);
            }
        }
    }
}
