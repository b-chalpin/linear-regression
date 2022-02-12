import time
from code_linear_regression.linear_regression import LinearRegression


def train_gd_model(X_train, y_train, X_test, y_test, max_degree, training_epochs, eta_list, lam_list):
    print("\n\n================================== GRADIENT DESCENT TRAINING ==================================")

    lr = LinearRegression()

    results = [] # results will hold dict of (degree, epochs, eta, lam, train_mse, test_mse, y_hat)

    for r in range(1, max_degree + 1):  # 1-based indexing
        print(f"degree {r}")
        
        print(f"\tepochs {training_epochs}")

        for eta_val in eta_list:
            print(f"\t\teta {eta_val}")

            for lam_val in lam_list:
                print(f"\t\t\tlam {lam_val}")

                start = time.time()
                train_mse, test_mse = lr.fit_metrics(X=X_train, y=y_train, X_test=X_test, y_test=y_test, epochs=training_epochs, eta=eta_val, degree=r, lam=lam_val)
                end = time.time()

                y_hat = lr.predict(X=X_test)
                
                min_train_mse = min(train_mse)
                min_test_mse = min(test_mse)

                result = {
                    "degree": r,
                    "epochs": training_epochs,
                    "eta": eta_val,
                    "lam": lam_val,
                    "train_mse": train_mse,
                    "test_mse": test_mse,
                    "min_train_mse": min_train_mse,
                    "min_train_mse_epoch": train_mse.index(min_train_mse),
                    "min_test_mse": min_test_mse,
                    "min_test_mse_epoch": test_mse.index(min_test_mse),
                    "y_hat": list(y_hat.flatten()), # json doesnt like the nd-array
                    "train_time": (end - start) # trainng time in seconds
                } 

                results.append(result)

                print(f"\t\t\tlam {lam_val} done;")

            print(f"\t\teta {eta_val} done;")

        print(f"\tepochs {training_epochs} done;")
            
        print(f"degree {r} done;")
            
    assert len(results) == max_degree * len(eta_list) * len(lam_list)
    print(f"\nnumber of training runs: {len(results)}")

    # add metadata
    training_output = {
        "metadata": {
            "max_degree": max_degree,
            "training_epochs": training_epochs,
            "eta_list": eta_list,
            "lam_list": lam_list
        },
        "results": results
    }

    return training_output
