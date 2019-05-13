
X_train, X_test, y_train, y_test = train_test_split(X,y,
        test_size=0.3)

# # use a full grid over all parameters -- for the random search
param_grid = {"max_depth": [3, None],
              "max_features": sp_randint(1, 25),
              "min_samples_split": sp_randint(1, 25),
              "min_samples_leaf": sp_randint(1, 10),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"],
              "n_estimators": sp_randint(5, 100)}

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    n_iter_search = 50
    clf = RandomizedSearchCV(RandomForestClassifier(),
            param_distributions=param_grid,
            n_iter=n_iter_search, cv=5, scoring='%s_weighted' % score,
            n_jobs=-1)

    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    # print()
    # for params, mean_score, scores in clf.grid_scores_:
    #     print("%0.3f (+/-%0.03f) for %r"
    #          % (mean_score, scores.std() * 2, params))
    print()
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
    print(confusion_matrix(y_true, y_pred))
