#include "parameter.h"
#include "function.h"

extern int     neighbors[number_of_computers][number_of_neighbors];
extern int     topology_basis[number_of_computers][2];
extern double  prob[number_of_states][number_of_states][number_of_states][number_of_states];
extern double  prob_real[number_of_states][number_of_states][number_of_states][number_of_states][number_of_computers];
extern int     basis[number_of_computers][scope];
extern int     type;

const int      number_of_type = 3;
double         count_state[number_of_type][number_of_states];
double         count_action[number_of_type][number_of_states];
const int      number_of_repeat = 25, number_of_iter = 200;
double         value_sum = 0, value_square_sum = 0, value_std, value_mean;

double action_expected_value(NumArray3D weight, const int* current_state, const int* current_action) {
    double curr_RHS = 0, prob_i, prob_j;
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
                curr_RHS += prob_i * prob_j * weight[state_i][state_j][k];
            }
        }
    }
    return curr_RHS;
}

double action_expected_value_true(NumArray3D weight, const int* current_state, const int* current_action) {
    double curr_RHS = 0, prob_i, prob_j;
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
                curr_RHS += prob_i * prob_j * weight[state_i][state_j][k];
            }
        }
    }
    return curr_RHS;
}


double find_best_action_enumerate(NumArray3D weight, const int* current_state, int* best_action) {
    int sum_action;
    int current_action[number_of_computers];
    for (int i = 0; i < number_of_computers; ++i) {
        current_action[i] = 0;
    }
    double best_value = -bigM, current_value;
    for (int i = 0; i < number_of_computers; i++) {
        current_action[i] = 1;
        current_value = action_expected_value(weight, current_state, current_action);
        if (current_value > best_value) {
            for (int j = 0; j < number_of_computers; ++j) {
                best_action[j] = current_action[j];
            }
            best_value = current_value;
        }
        current_action[i] = 0;
    }

    return best_value;

}

double find_best_action_enumerate_robust(NumArray3D& weight, const int* current_state, int* best_action) {
    int sum_action;
    NumArray3D prob_sa;
    int current_action[number_of_computers];
    for (int i = 0; i < number_of_computers; ++i) {
        current_action[i] = 0;
    }
    double best_value = -bigM, current_value;
    for (int i = 0; i < number_of_computers; i++) {
        current_action[i] = 1;
        current_value = transprob_lp(weight, prob_sa, current_state, current_action);
        if (current_value > best_value) {
            for (int j = 0; j < number_of_computers; ++j) {
                best_action[j] = current_action[j];
            }
            best_value = current_value;
        }
        current_action[i] = 0;
    }

    return best_value;

}

double find_best_action_enumerate_true(NumArray3D weight, const int* current_state, int* best_action) {
    int sum_action;
    int current_action[number_of_computers];
    for (int i = 0; i < number_of_computers; ++i) {
        current_action[i] = 0;
    }
    double best_value = -bigM, current_value;
    for (int i = 0; i < number_of_computers; i++) {
        current_action[i] = 1;
        current_value = action_expected_value_true(weight, current_state, current_action);
        if (current_value > best_value) {
            for (int j = 0; j < number_of_computers; ++j) {
                best_action[j] = current_action[j];
            }
            best_value = current_value;
        }
        current_action[i] = 0;
    }

    return best_value;

}

double find_best_action_enumerate_two(NumArray3D weight, const int* current_state, int* best_action) {
    int sum_action;
    int current_action[number_of_computers];
    for (int i = 0; i < number_of_computers; ++i) {
        current_action[i] = 0;
    }
    double best_value = -bigM, current_value;
    for (int i = 0; i < number_of_computers; i++) {
        current_action[i] = 1;
        for (int k = i + 1; k < number_of_computers; k++) {
            current_action[k] = 1;
            current_value = action_expected_value(weight, current_state, current_action);
            if (current_value > best_value) {
                for (int j = 0; j < number_of_computers; ++j) {
                    best_action[j] = current_action[j];
                }
                best_value = current_value;
            }
            current_action[k] = 0;
        }

        current_action[i] = 0;
    }

    return best_value;

}

double find_best_action_enumerate_two_true(NumArray3D weight, const int* current_state, int* best_action) {
    int sum_action;
    int current_action[number_of_computers];
    for (int i = 0; i < number_of_computers; ++i) {
        current_action[i] = 0;
    }
    double best_value = -bigM, current_value;
    for (int i = 0; i < number_of_computers; i++) {
        current_action[i] = 1;
        for (int k = i + 1; k < number_of_computers; k++) {
            current_action[k] = 1;
            current_value = action_expected_value_true(weight, current_state, current_action);
            if (current_value > best_value) {
                for (int j = 0; j < number_of_computers; ++j) {
                    best_action[j] = current_action[j];
                }
                best_value = current_value;
            }
            current_action[k] = 0;
        }

        current_action[i] = 0;
    }

    return best_value;

}

double find_best_action_enumerate_two_robust(NumArray3D& weight, const int* current_state, int* best_action) {
    int sum_action;
    NumArray3D prob_sa;
    int current_action[number_of_computers];
    for (int i = 0; i < number_of_computers; ++i) {
        current_action[i] = 0;
    }
    double best_value = -bigM, current_value;
    for (int i = 0; i < number_of_computers; i++) {
        current_action[i] = 1;
        for (int k = i + 1; k < number_of_computers; k++) {
            current_action[k] = 1;
            current_value = transprob_lp(weight, prob_sa, current_state, current_action);
            if (current_value > best_value) {
                for (int j = 0; j < number_of_computers; ++j) {
                    best_action[j] = current_action[j];
                }
                best_value = current_value;
            }
            current_action[k] = 0;
        }

        current_action[i] = 0;
    }

    return best_value;

}

double find_best_action(NumArray3D weight, const int* current_state, int* best_action) {
    try {
        GRBEnv env = GRBEnv();
        GRBModel model = GRBModel(env);

        // Create variables
        GRBVar* action = model.addVars(number_of_computers, GRB_BINARY);

        NumVarAction xi;
        int k, L;
        for (int ai = 0; ai < 2; ai++) {
            for (int aj = 0; aj < 2; aj++) {
                for (int index = 0; index < number_of_computers; index++) {
                    k = basis[index][0];
                    L = basis[index][1];
                    xi[ai][aj][index] = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS);
                    model.addConstr(xi[ai][aj][index] <= (2 * ai - 1) * action[k] + 1 - ai);
                    model.addConstr(xi[ai][aj][index] <= (2 * aj - 1) * action[L] + 1 - aj);
                    model.addConstr(xi[ai][aj][index] >= 1 + (2 * ai - 1) * (action[k] - ai) + (2 * aj - 1) * (action[L] - aj));
                }
            }
        }

        GRBLinExpr sum_action = 0;
        for (int k = 0; k < number_of_computers; ++k)
            sum_action += action[k];
        model.addConstr(sum_action == number_of_actions);

        double prob_i, prob_j;
        int index_k1, index_k2, index_L1, index_L2;
        GRBLinExpr obj_func = 0;
        for (int i0 = 0; i0 < number_of_states; i0++) {
            for (int j0 = 0; j0 < number_of_states; j0++) {
                for (int ai = 0; ai < 2; ai++) {
                    for (int aj = 0; aj < 2; aj++) {
                        for (int index = 0; index < number_of_computers; index++) {
                            k = basis[index][0];
                            L = basis[index][1];
                            if (ai == 1) {
                                if (i0 == number_of_states - 1) prob_i = 1.0;
                                else prob_i = 0.0;
                            }
                            else {
                                prob_i = prob[i0][current_state[k]][current_state[neighbors[k][0]]][current_state[neighbors[k][1]]];
                            }
                            if (aj == 1) {
                                if (j0 == number_of_states - 1) prob_j = 1.0;
                                else prob_j = 0.0;
                            }
                            else {
                                prob_j = prob[j0][current_state[L]][current_state[neighbors[L][0]]][current_state[neighbors[L][1]]];
                            }
                            obj_func += prob_i * prob_j * weight[i0][j0][index] * xi[ai][aj][index];
                        }
                    }
                }
            }
        }

        model.setObjective(obj_func, GRB_MAXIMIZE);
        model.set(GRB_IntParam_Threads, number_of_Threads);
        model.set(GRB_IntParam_OutputFlag, 0);
        model.set(GRB_IntParam_Cuts, 1);
        model.optimize();

        // Get the optimal solution
        for (int i = 0; i < number_of_computers; i++) {
            if (action[i].get(GRB_DoubleAttr_X) > 0.5) best_action[i] = 1;
            else best_action[i] = 0;
        }

        double value = model.get(GRB_DoubleAttr_ObjVal);
        return value;
    }
    catch (GRBException e) {
        cout << "Error code = " << e.getErrorCode() << endl;
        cout << e.getMessage() << endl;
        return 0;
    }
}

void monto_carlo_MIP_Thread(NumArray3D weight, const int* initial_state) {

    double random_number, value_opt = 0, value_k = 0, value, random_number_gen, random_number_cor;
    int current_state[number_of_computers], next_state[number_of_computers], action_opt[number_of_computers];
    for (int repeat = 0; repeat < number_of_repeat * number_of_Threads; ++repeat) {
        for (int i = 0; i < number_of_computers; i++) {
            current_state[i] = initial_state[i];
        }

        //generate_state_action(current_state, action_opt);
        value = 0;
        for (int iter = 0; iter < number_of_iter; ++iter) {
            for (int i = 0; i < number_of_computers; ++i)
                value += current_state[i] * pow(discount_factor, iter);

            if (number_of_actions == 1) find_best_action_enumerate(weight, current_state, action_opt);
            else find_best_action_enumerate_two(weight, current_state, action_opt);

            random_number_gen = ((double)rand() / (RAND_MAX));
            random_number_cor = ((double)rand() / (RAND_MAX));
            for (int i = 0; i < number_of_computers; ++i) {
                if (random_number_cor < correlation) random_number = random_number_gen;
                else random_number = ((double)rand() / (RAND_MAX));
                if (action_opt[i] == 1) next_state[i] = 2;
                else {
                    if (random_number < prob_real[0][current_state[i]][current_state[neighbors[i][0]]][current_state[neighbors[i][1]]][i])
                        next_state[i] = 0;
                    else if (random_number > 1 - prob_real[2][current_state[i]][current_state[neighbors[i][0]]][current_state[neighbors[i][1]]][i])
                        next_state[i] = 2;
                    else next_state[i] = 1;
                }
            }
            for (int i = 0; i < number_of_computers; i++) {
                current_state[i] = next_state[i];
            }
        }
        value_sum += value;
        value_square_sum += value * value;
    }
}

void monto_carlo_MIP_Thread_true(NumArray3D weight, const int* initial_state) {

    double random_number, value_opt = 0, value_k = 0, value, random_number_gen, random_number_cor;
    int current_state[number_of_computers], next_state[number_of_computers], action_opt[number_of_computers];
    for (int repeat = 0; repeat < number_of_repeat * number_of_Threads; ++repeat) {
        for (int i = 0; i < number_of_computers; i++) {
            current_state[i] = initial_state[i];
        }

        //generate_state_action(current_state, action_opt);
        value = 0;
        for (int iter = 0; iter < number_of_iter; ++iter) {
            for (int i = 0; i < number_of_computers; ++i)
                value += current_state[i] * pow(discount_factor, iter);

            if (number_of_actions == 1) find_best_action_enumerate_true(weight, current_state, action_opt);
            else find_best_action_enumerate_two_true(weight, current_state, action_opt);

            random_number_gen = ((double)rand() / (RAND_MAX));
            random_number_cor = ((double)rand() / (RAND_MAX));
            for (int i = 0; i < number_of_computers; ++i) {
                if (random_number_cor < correlation) random_number = random_number_gen;
                else random_number = ((double)rand() / (RAND_MAX));
                if (action_opt[i] == 1) next_state[i] = 2;
                else {
                    if (random_number < prob_real[0][current_state[i]][current_state[neighbors[i][0]]][current_state[neighbors[i][1]]][i])
                        next_state[i] = 0;
                    else if (random_number > 1 - prob_real[2][current_state[i]][current_state[neighbors[i][0]]][current_state[neighbors[i][1]]][i])
                        next_state[i] = 2;
                    else next_state[i] = 1;
                }
            }
            for (int i = 0; i < number_of_computers; i++) {
                current_state[i] = next_state[i];
            }
        }
        value_sum += value;
        value_square_sum += value * value;
    }
}

void monto_carlo_MIP_Thread_robust(NumArray3D& weight, const int* initial_state) {

    double random_number, value_opt = 0, value_k = 0, value, random_number_gen, random_number_cor;
    int current_state[number_of_computers], next_state[number_of_computers], action_opt[number_of_computers];
    for (int repeat = 0; repeat < number_of_repeat * number_of_Threads; ++repeat) {
        for (int i = 0; i < number_of_computers; i++) {
            current_state[i] = initial_state[i];
        }

        //generate_state_action(current_state, action_opt);
        value = 0;
        for (int iter = 0; iter < number_of_iter; ++iter) {
            for (int i = 0; i < number_of_computers; ++i)
                value += current_state[i] * pow(discount_factor, iter);

            if (number_of_actions == 1) find_best_action_enumerate_robust(weight, current_state, action_opt);
            else find_best_action_enumerate_two_robust(weight, current_state, action_opt);

            random_number_gen = ((double)rand() / (RAND_MAX));
            random_number_cor = ((double)rand() / (RAND_MAX));
            for (int i = 0; i < number_of_computers; ++i) {
                if (random_number_cor < correlation) random_number = random_number_gen;
                else random_number = ((double)rand() / (RAND_MAX));
                if (action_opt[i] == 1) next_state[i] = 2;
                else {
                    if (random_number < prob_real[0][current_state[i]][current_state[neighbors[i][0]]][current_state[neighbors[i][1]]][i])
                        next_state[i] = 0;
                    else if (random_number > 1 - prob_real[2][current_state[i]][current_state[neighbors[i][0]]][current_state[neighbors[i][1]]][i])
                        next_state[i] = 2;
                    else next_state[i] = 1;
                }
            }
            for (int i = 0; i < number_of_computers; i++) {
                current_state[i] = next_state[i];
            }
        }
        value_sum += value;
        value_square_sum += value * value;
    }
}

double monto_carlo_MIP(NumArray3D weight, const int* initial_state, const double upper_bound) {

    /*std::vector<std::thread> threads(number_of_Threads);

    for (int i = 0; i < number_of_Threads; ++i) {
        threads[i] = std::thread(monto_carlo_MIP_Thread, weight, initial_state);
    }

    for (auto& thread : threads) {
        thread.join();
    }*/
    monto_carlo_MIP_Thread(weight, initial_state);
    double total_repeat = number_of_Threads * number_of_repeat;
    value_mean = value_sum / total_repeat;
    value_std = sqrt(value_square_sum / total_repeat - value_mean * value_mean);

    return value_mean;
}

double monto_carlo_MIP_true(NumArray3D weight, const int* initial_state, const double upper_bound) {

    /*std::vector<std::thread> threads(number_of_Threads);

    for (int i = 0; i < number_of_Threads; ++i) {
        threads[i] = std::thread(monto_carlo_MIP_Thread, weight, initial_state);
    }

    for (auto& thread : threads) {
        thread.join();
    }*/
    monto_carlo_MIP_Thread_true(weight, initial_state);
    double total_repeat = number_of_Threads * number_of_repeat;
    value_mean = value_sum / total_repeat;
    value_std = sqrt(value_square_sum / total_repeat - value_mean * value_mean);

    return value_mean;
}

double monto_carlo_MIP_robust(NumArray3D& weight, const int* initial_state, const double upper_bound) {

    /*std::vector<std::thread> threads(number_of_Threads);

    for (int i = 0; i < number_of_Threads; ++i) {
        threads[i] = std::thread(monto_carlo_MIP_Thread, weight, initial_state);
    }

    for (auto& thread : threads) {
        thread.join();
    }*/
    monto_carlo_MIP_Thread_robust(weight, initial_state);
    double total_repeat = number_of_Threads * number_of_repeat;
    value_mean = value_sum / total_repeat;
    value_std = sqrt(value_square_sum / total_repeat - value_mean * value_mean);

    return value_mean;
}

