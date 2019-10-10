
import math, random, re, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from statistics import mode
from sklearn.metrics import roc_curve, auc, r2_score, confusion_matrix
from sklearn.metrics import log_loss
# ignoring warnings for the time being
import warnings
warnings.filterwarnings('ignore')


def get_titles(df):
    '''
    Creating function to get titles from names. this will be used later to impute ages in train and test.
    :param df:
    :return: a list of titles
    '''
    titles = []
    title = ''
    for i in range(len(df.Name)):
        if type(df.Name[i]) != str:  # check to make sure that the function is being passed a string, if passed a nan or a number the name will be set to a blank string
            df.Name[i] = ''
        split_name = re.split('[, .]', df.Name[i])
        for j in range(1, len(split_name) - 1):
            if split_name[j - 1] == '' and split_name[j + 1] == '':
                title = split_name[j]
        titles.append(title)
    return titles


def mean_imputing_ages(df, df_encode, df_keep, titles):
    '''
    function to imput ages, ages are imputed from the mean age of all people with the same title.
    :param df:
    :param df_encode:
    :param df_keep:
    :param titles:
    :return: N/a
    '''
    mean_age = df.Age.mean()
    for i in range(len(pd.value_counts(titles))):
        age = np.empty(df_encode.Title.value_counts()[i], dtype=float)
        j = 0
        for x in range(len(df)):
            if df_encode.Title[x] == str(df_encode.Title.value_counts().index[i]):
                age[j] = df_keep.Age[x]
                j += 1
        age = age[~np.isnan(age)]
        mu = age.mean()
        j = 0
        for x in range(len(df)):
            if df_encode.Title[x] == str(df_encode.Title.value_counts().index[i]) and np.isnan(df_keep.Age[x]):
                df_keep.Age[x] = mu
                j += 1
    for j in range(len(df)):
        if np.isnan(df_keep.Age[j]):
            df_keep.Age[j] = mean_age


def impute_Pclass(df_train, df_test, df_test_encode, mean_first, mean_second, mean_third):
    '''
    check to see if there are any missing values, if so, the fare is used to find what class they may have been in.
    :param df_train:
    :param df_test:
    :return: N/a
    '''
    for row in range(df_test.shape[0]):
        dis1, dis2, dis3 = 0, 0, 0
        if np.isnan(df_test_encode.Pclass[row]):
            if ~np.isnan(df_test.Fare[row]):
                dis1 = abs(mean_first - df_test.Fare.iloc[row])
                dis2 = abs(mean_second - df_test.Fare.iloc[row])
                dis3 = abs(mean_third - df_test.Fare.iloc[row])
                if (dis1 < dis2) and (dis1 < dis3):
                    df_test_encode.Pclass[row] = 1
                elif dis2 < dis1 and dis2 < dis3:
                    df_test_encode.Pclass[row] = 2
                elif dis3 < dis2 and dis3 < dis1:
                    df_test_encode.Pclass[row] = 3
            else:
                df_test_encode.Pclass[row] = mode(df_test.Pclass)


def impute_Sex(df_test, df_test_encode, df_train, titles_train, titles_test):
    '''
    function will impute sex if the given data frame has nan's in it for the feature Sex
    :return: N/a
    '''
    indices_missing_sex = df_test.index[df_test_encode.Sex.isnull()]  # these are all the indices of where the sex is a nan
    j = 0
    for title in df_test_encode.Title[indices_missing_sex]:  # $$alternative$$ for title in df_test_encode.Title[df_test.Sex.isnull()]:
        if titles_train.count(title) > 1:  # check to make sure that there is more than one of that title in titles_train
            indices = [i for i, x in enumerate(titles_train) if x == title]  # find all the indices in titles_train where the title that we are currently looking at (title that is missing), that match the title.
            df_test_encode.Sex[indices_missing_sex[j]] = mode([df_train.Sex[index] for index in indices])  # finding the mode sex for all of the title that we are currently on and imputing that sex
        elif titles_test.count(title) > 1:  # check to make sure that there is more than one of that title in title_test
            indices = [i for i, x in enumerate(titles_test) if x == title]
            df_test_encode.Sex[indices_missing_sex[j]] = mode([df_test.Sex[index] for index in indices])
        elif len(title) == 0 or title[-1] == 'a' or title[-1] == 's' or title[-1] == 'e' or title[-1] == 'u':
            df_test_encode.Sex[indices_missing_sex[j]] = 'female'
        else:
            df_test_encode.Sex[indices_missing_sex[j]] = 'male'
        j += 1


def impute_SibSp(df_test, df_test_keep, df_train):
    for row in range(df_test.shape[0]):
        if np.isnan(df_test_keep.SibSp[row]):
              df_test_keep.SibSp[row] = mode(df_train.SibSp)


def impute_Parch(df_test, df_test_keep, df_train):
    for row in range(df_test.shape[0]):
        if np.isnan(df_test_keep.Parch[row]):
              df_test_keep.Parch[row] = mode(df_train.Parch)


def impute_Fare(df_test_keep, df_test_encode, mean_first, mean_second, mean_third):
    '''
    function to impute missing value from Fare. a check is done to see what class the passenger is in and the mean fare from this class is given to the passenger.
    :param df_test_keep: this is the data frame that holds the fare value
    :param df_test_encode: this is used to find out what class the passenger with the missing fare is in
    :param mean_first: this is the mean fare value of the passengers in the first class, taken from the training data frame
    :param mean_second: this is the mean fare value of the passengers in the secnond class, taken from the training data frame
    :param mean_third: this is the mean fare value of the passengers in the third class, taken from the training data frame
    :return: N/a
    '''
    for row in range(df_test_keep.shape[0]):
        if np.isnan(df_test_keep.Fare[row]):
            if df_test_encode.Pclass[row] == 1:  # passenger is in first class
                df_test_keep.Fare[row] = mean_first
            elif df_test_encode.Pclass[row] == 2:  # passenger is in second class
                df_test_keep.Fare[row] = mean_second
            else:  # passenger is in third class, also catches anything strange that is happening and isn't already caught
                df_test_keep.Fare[row] = mean_third


def nans_in_data_frame(dataframe):
    '''
    a sound will be emitted if there are any null values in the data frame that you are passing through
    :param dataframe: this is the dataframe that you want to check to make sure that there are no null values.
    :return: N/a
    '''
    if dataframe.isnull().any().any() == True:
        print('You have got nans somewhere in the data frame')
        import winsound
        duration = 500 # in mili secs
        freq = 10000 #Hz
        winsound.Beep(freq, duration)


def get_folds(K, Xy_train):
    '''
    function to obtain our folds, once obtained we can do our cross validation
    :param K: Number of folds that you want
    :param X_train: training data set
    :return: a data frame where each fold is a column in the data frame, and the number of folds
    '''
    fold_size = round(len(Xy_train)/K)  # defining the length of the fold (rounding down)
    # creating a data frame of nans of size(row:biggest fold size, column: number of folds)
    df_rand = pd.DataFrame(pd.np.empty([fold_size+(K-1), K])*pd.np.nan)
    df_folds = pd.DataFrame()  # this is what we will be returning, each column is one fold
    arr = []
    for j in range(0, len(Xy_train)):
        # populating an array that we will use for our indices
        arr.append(j)
    for k in range(0, K-1):
        # creating a data frame of nans of size (row:fold size, no. of columns: 1)
        # df fold is essentially a filing cabinet where each cell will contain an array from x train
        df_fold = pd.DataFrame(pd.np.empty([fold_size, 1])*pd.np.nan).astype(object)
        for i in range(0, fold_size):
            rand_ind = random.randint(0, len(arr)-1)  # index of arr
            rand_num = arr[rand_ind]
            df_rand.iloc[i,  k] = rand_num  # putting the number into df rand
            # put the random number from
            df_fold.at[i, 0] = np.array(Xy_train.iloc[int(df_rand.iloc[i, k])])
            arr.pop(rand_ind)  # popping from arr so that it can't be chosen again
        df_folds = pd.concat([df_folds, df_fold], axis=1)
    df_arr = pd.DataFrame(arr)  # adding what's left in the array arr to a new data frame
    for l in range(0, len(df_arr)):
        # creating the last filing cabinet
        df_fold.at[l, 0] = np.array(Xy_train.iloc[int(df_arr.iloc[l, 0])])
    df_folds = pd.concat([df_folds, df_fold], axis=1)  # adding the last filing cabinet to our wall of filing cabinets
    df_folds.columns = (str(h) for h in range(0, K))
    return df_folds, K


def manual_confusion_matrix(y_actu, y_pred):
    '''
    this function simply compared the values that our model predicts with the actual values.
    it outputs 4 values, TP, FP, FN, TN
    '''
    tp = fp = tn = fn = 0
    if len(y_actu) != len(y_pred):
        print('error: can not compare things that are not the same length')
    for label in range(len(y_actu)):
        if y_actu.iloc[label, 0] == y_pred[label] and y_actu.iloc[label, 0] == 0:
            tn += 1
        elif y_actu.iloc[label, 0] != y_pred[label] and y_actu.iloc[label, 0] == 1:
            fn += 1
        elif y_actu.iloc[label, 0] != y_pred[label] and y_actu.iloc[label, 0] == 0:
            fp += 1
        elif y_actu.iloc[label, 0] == y_pred[label] and y_actu.iloc[label, 0] == 1:
            tp += 1
        else:
            print('error: something is wrong in the comparison')
    return tn, fp, fn, tp


def model_fit_predict(algo, X_train, y_train, X_test):
    start_time = time.time()
    algo.fit(X_train, y_train)
    log_time = time.time() - start_time
    y_pred = algo.predict(X_test)
    return log_time, y_pred


def cross_validation(K, algo, df_folds):
    '''
    function to perform cross validation on the data frame, this is written so that we can access and understand the different folds. allows us to the data frame in further detail and see how different processes and procedures is affecting our model.
    :param K: the number of folds
    :param X_train: the training data frame that we are passing
    :param algo: the type of algorithm the we are passing.
    :return:
    '''
    df_conmat = pd.DataFrame(pd.np.empty([K, 4])*pd.np.nan)  # creating a data frame for the confusion matrices
    df_conmat.columns = ['TrueNeg', 'FalsePos', 'FalseNeg', 'TruePos']  # naming the columns
    X_train_fold_total = pd.DataFrame()
    y_train_fold_total = pd.DataFrame()
    X_test_fold_total = pd.DataFrame()
    y_test_fold_total = pd.DataFrame()
    logloss_fold_total = []
    log_time_fold_total = []
    r2_fold_total = []
    for i in range(0, K):
        df_folds_copy = df_folds  # each time we change folding set up (loop through i), we create a copy of initial wall of filing cabinets
        test_fold = df_folds_copy.iloc[:, i].to_frame()  # one particular filing cabinet (i) held out for testing purposes. Making it a daraframe as this allows everything to work properly below this point
        x_test_fold = test_fold
        y_test_fold = pd.DataFrame(pd.np.empty([len(test_fold), 1]) * pd.np.nan).astype(object)  # creating df to populate in j loop
        for j in range(len(test_fold.dropna())):  # looping through each drawer(aka array)in filing cabinet (nan check is necessary, because the last filing cabinet (fold) is longer and dictates the size of the other filing cabinets, thus the first K-1 folds have at least one nan at the bottom) Note: len(test_fold[~test_fold.isna()]) won't work for some reason:
            y_test_fold.iloc[j, 0] = test_fold.iloc[j, 0][0]  # add the very first file (element) in each drawer (array) to y_test_fold
            x_test_fold.iloc[j, 0] = np.delete(x_test_fold.iloc[j, 0], 0)  # remove the first file from each drawer to yield x_test_fold
        # Do similar for train_fold (outside of j loop, back in i loop)
        train_fold = df_folds_copy  # creating the data frame of all the training folds
        train_fold = train_fold.drop(train_fold.columns[i], axis=1)  # dropping the test filing cabinet (i) from the wall. Using this for 1) size of x_train_fold and 2) also maybe when fitting models
        train_fold_melted = train_fold.melt().dropna().value.to_frame()  # collapsing all columns into one long col, dropping nan's, row index doesn't update
        train_fold_melted.reset_index(drop=True, inplace=True)  # have to reset the indices as they didn't update after dropping nans
        x_train_fold = train_fold_melted
        y_train_fold = pd.DataFrame(pd.np.empty([len(train_fold_melted), 1]) * pd.np.nan).astype(object)
        for j in range(len(train_fold_melted)):
            y_train_fold.iloc[j, 0] = train_fold_melted.iloc[j, 0][0]
            x_train_fold.iloc[j, 0] = np.delete(x_train_fold.iloc[j, 0], 0)
        # transforming our data so that it's in the proper format to fit it to an algorithm.
        # this should propably have been set up better earlier so that we don't have to do this.
        X_train_fold = pd.DataFrame(pd.np.empty([len(x_train_fold), len(x_train_fold.iloc[0, 0])])*pd.np.nan)
        for drawer in range(0, len(x_train_fold)):
            for file in range(0, len(x_train_fold.iloc[0, 0])):
                X_train_fold.iloc[drawer, file] = x_train_fold.iloc[drawer, 0][file]
        # transforming our data so that it's in the proper format to fit it to an algorithm.
        # also need to get rid of the nans as these are floats and don't work.
        # once again this could have been done earlier so that we don't have to do this now.
        X_test_fold = pd.DataFrame(pd.np.empty([len(x_test_fold.dropna()), len(x_test_fold.iloc[0, 0])])*pd.np.nan)
        for drawer in range(len(x_test_fold.dropna())):
            for file in range(len(x_test_fold.iloc[0, 0])):
                X_test_fold.iloc[drawer, file] = x_test_fold.iloc[drawer, 0][file]
        # making sure that we've gotten rid of all our nans
        # clean this up later as some data frames won't have any nans
        X_test_fold = X_test_fold.dropna()
        y_test_fold = y_test_fold.dropna()
        y_test_fold = y_test_fold.convert_objects(convert_numeric=True)
        # without the above line we get the following error after fitting our algorithm
        # Classification metrics can't handle a mix of unknown and binary targets
        # look into later,  everything in the ytestfold df is an object and need to be in int or float
        model = model_fit_predict(algo, X_train_fold, y_train_fold, X_test_fold)  # fitting algo to current combination of folds
        log_time_fold = model[0]
        log_time_fold_total.append(log_time_fold)
        y_pred_fold = model[1]
        conmat = manual_confusion_matrix(y_test_fold, y_pred_fold)
        # populating the table of confusion matrices
        for val in range(len(conmat)):
            df_conmat.iloc[i, 0] = conmat[0]
            df_conmat.iloc[i, 1] = conmat[1]
            df_conmat.iloc[i, 2] = conmat[2]
            df_conmat.iloc[i, 3] = conmat[3]
        # appending data frame to total data frame for further analysis
        X_train_fold_total = pd.concat([X_train_fold_total, X_train_fold], axis=1)
        y_train_fold_total = pd.concat([y_train_fold_total, y_train_fold], axis=1)
        X_test_fold_total = pd.concat([X_test_fold_total, X_test_fold], axis=1)
        y_test_fold_total = pd.concat([y_test_fold_total, y_test_fold], axis=1)
        logloss_fold = log_loss(y_test_fold, algo.predict_proba(X_test_fold)[:, 1])  # the log loss is found for each fold that we do
        logloss_fold_total.append(logloss_fold)
        r2_fold = r2_score(y_test_fold, algo.predict_proba(X_test_fold)[:, 1])  # the log loss is found for each fold that we do
        r2_fold_total.append(r2_fold)
    return df_conmat, X_train_fold_total, y_train_fold_total, X_test_fold_total, y_test_fold_total, K, logloss_fold_total, np.mean(log_time_fold_total)  #X_train_fold_total, y_train_fold_total, X_test_fold_total, y_test_fold_total, K, df_conmat, r2_fold_total, logloss_fold_total, np.mean(log_time_fold_total) # y_pred_fold (no need to return, this is fed into the manual confusion matrix function so we have gotten what we need from it)


def rotate(x, y, xo, yo, theta):  # rotating our frame of reference so that we are on the zero skilled line(y=x).
    xr = math.cos(theta)*(x-xo)-math.sin(theta)*(y-yo) + xo
    yr = math.sin(math.radians(theta))*(x-xo)+math.cos(math.radians(theta))*(y-yo) + yo
    return [xr, yr]


def get_threshold(thresholds, fpr, tpr):
    max_height = 0
    for i in range(len(thresholds)):
        threshold_height = rotate(fpr[i], tpr[i], 0, 0, -45)[1]
        if threshold_height > max_height:
            max_height = threshold_height
            threshold = thresholds[i]
    return threshold


def plot_roc(thresholds, roc_auc, fpr, tpr):
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc, marker='|')
    #for l, txt in enumerate(thresholds):
     #   plt.annotate(txt, (fpr[l], tpr[l]))
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('ROC Curve')
    plt.show()


def assess_roc(X_train_fold_total, X_test_fold_total, y_train_fold_total, y_test_fold_total, K, algo):
    threshold_list = []
    # y_pred_prob_df = pd.DataFrame() # leaving this here as the check
    i = j = 0
    x_train_fold_width = int(X_train_fold_total.shape[1]/K) # the amount we want to increment by through the big dataframe
    plt.figure(figsize=(8,8))
    while i < X_train_fold_total.shape[1]:
        # https://stackoverflow.com/questions/31417487/sklearn-logisticregression-and-changing-the-default-threshold-for-classification
        # in training we edit the weights based on a set threshold, in our ROC curve we change our threshold based on
        # the set weights from training. this allows us to find our 'optimal' threshold, and we already have our weights.
        # this can only be done in a cross validation loop. if it wasn't done in cross_validation (see your man's second point)
        algo_fit = algo.fit(X_train_fold_total.iloc[:, i:i+x_train_fold_width].dropna(), y_train_fold_total.iloc[:, j].dropna())
        y_pred_prob = algo_fit.predict_proba(X_test_fold_total.iloc[:, i:i+x_train_fold_width].dropna())[::, 1]
        fpr, tpr, thresholds = roc_curve(y_test_fold_total.iloc[:, j].dropna().to_numpy().astype(int), y_pred_prob)  # drop_intermediate = False # np.unique(y_pred_prob), will give us the number of thresholds
        # what does ROC do? it give a prediction based off predicted probabilities, it goes through each of these
        # probabilities and iteratively sets that probability as the correct threshold and finds the difference
        # between the tpr and the fpr.
        # the best threshold is the one that gives the greatest difference between the tpr and the fpr.
        roc_auc = auc(fpr, tpr)
        plot_roc(thresholds, roc_auc, fpr, tpr)
        # https://stats.stackexchange.com/questions/186337/average-roc-for-repeated-10-fold-cross-validation-with-probability-estimates
        # that link has a lovely lovely lovely!!! plot
        threshold_list.append(get_threshold(thresholds, fpr, tpr))
        # y_pred_prob_df = pd.concat([y_pred_prob_df, pd.DataFrame(y_pred_prob)], axis = 1) # leaving this here as the check
        i += x_train_fold_width
        j += 1
    return roc_auc, threshold_list, fpr, tpr, thresholds
    # we can also return len(np.unique(y_pred_prob)) to find our how many unique values we have compared to len(y_pred_prob)
    # ## https://stackoverflow.com/questions/31417487/sklearn-logisticregression-and-changing-the-default-threshold-for-classification
    # so what is our best threshold? the best threshold is the best trade off we have between trp and fpr. it is the point
    # on the roc curve that is furthest away from the unskilled line. it is our best trade off between trp and fpr (ie the
    # biggest difference between the two.
    # https://stackoverflow.com/questions/31417487/sklearn-logisticregression-and-changing-the-default-threshold-for-classification?rq=1
    # best_thresh = mode(resultROC)


def model_metrics(confusion_matrix_table):
    '''
    output the statistical measures of our model. mean values are take from cross validation and are calculated from K folds.
    :param confusion_matrix_table: taken from cross validation function, it is the TP, TN, FP, and FN of each folding iteration
    :return:    accuracy (how accurate the model is, using TP, TN, FP, and FN)
                positive predictive value/precision (how often a value represents a positive value),
                sensitivity/recall/TPR (percentage correctly found),
                specificity/1-FPR (percentage correctly rejected)
                f1score (trade off between precision and recall)
    '''
    # accuracy
    acc = (confusion_matrix_table['TruePos'].mean() + confusion_matrix_table['TrueNeg'].mean())\
          / (confusion_matrix_table['TruePos'].mean() + confusion_matrix_table['TrueNeg'].mean()
            + confusion_matrix_table['FalsePos'].mean() + confusion_matrix_table['FalseNeg'].mean())
    # precision (positive predictive value)
    precision = (confusion_matrix_table['TruePos'].mean())/(confusion_matrix_table['TruePos'].mean() + confusion_matrix_table['FalsePos'].mean())
    # sensitivity (recall) (TPR)
    sens = (confusion_matrix_table['TruePos'].mean())/(confusion_matrix_table['TruePos'].mean() + confusion_matrix_table['FalseNeg'].mean())
    # specificity (1-FPR)
    specif = (confusion_matrix_table['TrueNeg'].mean())/(confusion_matrix_table['TrueNeg'].mean() + confusion_matrix_table['FalsePos'].mean())
    # F1Score
    f1score = 2*(precision*sens)/(precision + sens)  # thia gives us our relationship between precision and recall (given by 2*(precision*recall)/(precision + recall))
    metrics_dict = {'acc': acc, 'prec': precision, 'sens':sens, 'spec':specif, 'f1':f1score}
    return metrics_dict  # 'acc, precision, sens, specif, f1score', acc, precision, sens, specif, f1score, metrics_dict


def algo_fit_output(algo, K, df_folds, output_train, X_pseudo_test, y_pseudo_test, j, no_threshold=True):
    '''
    performs cross validation, model metrics and assess roc and presents it in a tidy manner.
    :param algo:
    :param K:
    :param df_folds:
    :param output:
    :param j:
    :param dict:
    :return:
    '''
    cross_val_output = cross_validation(K, algo, df_folds)
    X_train_fold_total, y_train_fold_total, X_test_fold_total, y_test_fold_total, K, logloss_fold_total, run_time_mean = [cross_val_output[i] for i in range(1, len(cross_val_output))]  # the range is starting at one here because the first output is confusion matrix data frame
    output_train.iloc[j, 0] = algo # .__class__  # this is naming
    if no_threshold:
        metrics_dict = model_metrics(cross_val_output[0])
        for index, key in enumerate(metrics_dict.keys()):
            if key == output_train.columns[index + 1]:
                output_train.iloc[j, index + 1] = metrics_dict.get(key)
        # df_conmat = model_metrics(cross_val_output_LR[0])[1:6]    # leaving here in case is useful/needed at some stage
        # result_roc = assess_roc(X_train_fold_total, X_test_fold_total, y_train_fold_total, y_test_fold_total, K, LR)  # leaving here in case is useful/needed at some stage
        # output.iloc[j, ] = model_metrics(cross_val_output[0])[1:6]  # indexing to get rid of strings at index [0]
        output_train.iloc[j, len(metrics_dict)+1:len(cross_val_output)] = [cross_val_output[i] for i in range(6, len(cross_val_output))]
        output_train.iloc[j, len(cross_val_output)] = assess_roc(X_train_fold_total, X_test_fold_total, y_train_fold_total, y_test_fold_total, K, algo)[0]
    else:

        output_pseudo = pd.DataFrame(np.empty([1, output_train.shape[1]]) * pd.np.nan,columns = ['model', 'acc', 'prec', 'sens', 'spec', 'f1', 'loss', 'run time', 'AUC', 'selected params']).astype(object)
        output_pseudo.iloc[j, 0] = algo # .__class__
        #output_pseudo.iloc[j, 0] = algo.__class__  # this is naming
        resultROC = assess_roc(X_train_fold_total, X_test_fold_total, y_train_fold_total, y_test_fold_total, K, algo)
        predprobas = resultROC[1]
        THRESHOLD = np.mean(predprobas)
        pseudo_preds = np.where(algo.predict_proba(X_pseudo_test)[:, 1] > THRESHOLD, 1, 0)
        pseudo_conmat = confusion_matrix(y_pseudo_test, pseudo_preds)
        df_pseudo_conmat = pd.DataFrame(pseudo_conmat.reshape((1,4)), columns=['TrueNeg','FalsePos','FalseNeg','TruePos'])
        metrics_dict = model_metrics(df_pseudo_conmat)
        for index, key in enumerate(metrics_dict.keys()):
            if key == output_pseudo.columns[index + 1]:
                output_pseudo.iloc[j, index + 1] = metrics_dict.get(key)
        output_pseudo.iloc[j, len(metrics_dict)+1:len(cross_val_output)] = [cross_val_output[i] for i in range(6, len(cross_val_output))]
        #output_pseudo.iloc[j, len(cross_val_output)] = assess_roc(X_train_fold_total, X_test_fold_total, y_train_fold_total, y_test_fold_total, K, algo)[0]
        #output_train.append(output_pseudo, ignore_index=True)
        output_train = pd.concat([output_train,output_pseudo], axis=0, ignore_index=True)
    return output_train


