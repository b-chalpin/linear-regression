import time
from code_linear_regression.linear_regression import LinearRegression
    
lr = LinearRegression()

results = [] # results will hold dict of (degree, epochs, eta, lam, train_mse, test_mse, y_hat)

def train_cf_model(X_train, y_train, X_test, y_test, max_degree, lam_list):
    print("\n\n================================== CLOSED FORM TRAINING ==================================")

    for r in range(1, max_degree + 1):  # 1-based indexing
        print(f"degree {r}")
        
        for lam_val in lam_list:
            print(f"\tlam {lam_val}")

            start = time.time()
            lr.fit(X=X_train, y=y_train, CF=True, lam=lam_val, degree=r)
            end = time.time()

            train_mse = lr.error(X=X_train, y=y_train)
            test_mse = lr.error(X=X_test, y=y_test)

            y_hat = lr.predict(X=X_test)

            result = {
                "degree": r,
                "lam": lam_val,
                "train_mse": train_mse,
                "test_mse": test_mse,
                "y_hat": list(y_hat.flatten()), # json doesnt like the nd-array
                "train_time": (end - start) # trainng time in seconds
            }

            results.append(result)

            print(f"\tlam {lam_val} done;")

        print(f"degree {r} done;")

    assert len(results) == max_degree * len(lam_list)
    print(f"\nnumber of training runs: {len(results)}")

    # add metadata
    training_output = {
        "metadata": {
            "max_degree": max_degree,
            "lam_list": lam_list
        },
        "results": results
    }

    return training_output
