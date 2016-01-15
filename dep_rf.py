# -*- coding: utf-8 -*-
"""
Deprecated patch testing functions that need to updated for the HOG featureset.
Created on Fri Jan 15 11:17:44 2016

@author: Sadi
"""

"""
Given access to the full set of existing SliceInfo objects, and an index which
represents a "test slice", generates training data using all the SliceInfo
objects and their features (except for the ith SliceInfo object) and trains
a random forest classifier on that data (see "generate_training_feats" and
"train_rf_classifier"). Then, asks the trained classifier to predict the
classes of the patches stored inside the test slice (the ith SliceInfo object).
Records whether the random forest correctly classified each slice, and then
displays the results. The predictions are only on the existing features stored
for training in the SliceInfo object, so it doesn't check every single patch.
Essentially, this means that it only tests on the PCPs of the ith SliceInfo.

**This function is not updated for the HOG featureset that is now added by
default to the SliceInfo class. Therefore, you'll get an error and the
program will crash if you try to call this function right now.
"""
def test_rf_feats(slice_infos, i):
    print("Testing Case {}/{}: ".format(i+1, len(slice_infos)), end="")
    
    # We need to train on masked and unmasked features from every
    # slice except for this one
    feats, labels = generate_training_feats(slice_infos, i)

    # Train the classifier
    RF = train_rf_classifier(feats, labels, no_trees)
    
    # Test the RF on each of the feats from the test slice
    test_sl = slice_infos[i]
    successes = 0
    trials = 0
    total = len(test_sl.feats_m) + len(test_sl.feats_n)
    for feat_m in test_sl.feats_m:
        # flatten the feature so it can be tested
        feat_m = feat_m.flatten().reshape(1, -1)
        if RF.predict(feat_m) == 'M':
            successes += 1
        trials += 1
        if(len(sys.argv) < 2):
            print("\rTesting Case {}/{}: {}/{}/{} successes.".format(
            i+1, len(slice_infos), successes, trials, total), end="")
    for feat_n in test_sl.feats_n:
        # flatten the feature so it can be tested
        feat_n = feat_n.flatten().reshape(1, -1)
        if RF.predict(feat_n) == 'N':
            successes += 1
        trials += 1
        if(len(sys.argv) < 2):
            print("\rTesting Case {}/{}: {}/{}/{} successes.".format(
            i+1, len(slice_infos), successes, trials, total), end="")
    
    print("")
    print(np.asarray(slice_infos[i].feats_m).shape)
    
    return successes, trials


"""
Works like test_rf_feats, but instead of testing the classifier on the PCPs
of each SliceInfo object, tests the classifier on *every* patch that is
generated from the slice image.

**This function is not updated for the HOG featureset that is now added by
default to the SliceInfo class. Therefore, you'll get an error and the
program will crash if you try to call this function right now.
"""
def test_rf_feats_fullspec(slice_infos, i):
    feats, labels = generate_training_feats(slice_infos, i)
    RF = train_rf_classifier(feats, labels, no_trees)
    
    test_sl = slice_infos[i]
    image = test_sl.slice_im
    label = test_sl.slice_lb
    patches_m, patches_n = extract_roi_patches(image, label, psize)
    patches_n = patches_n
    plabels_m = ['M' for m in patches_m]
    plabels_n = ['N' for n in patches_n]
    kernels = generate_kernels()
    
    tot = len(patches_m) + len(patches_n)
    
    res1 = []
    res2 = []
    
    t0 = time()
    
    if len(sys.argv) >= 2:
        res1 = Parallel(n_jobs=int(sys.argv[1]))(delayed(check_classify)(RF, kernels, p, plabels_m[i], i, tot) for i, p in enumerate(patches_m))
        res2 = Parallel(n_jobs=int(sys.argv[1]))(delayed(check_classify)(RF, kernels, p, plabels_n[i], i, tot) for i, p in enumerate(patches_n))
    else:
        for i,p in enumerate(patches_m): # go through each patch, and classify it!
            res1.append(check_classify(RF, kernels, p, plabels_m[i], i, tot))
        pass

    dt = time() - t0    
    
    print(res1.count(True) + res2.count(True))
    print("Finished in {:.2f} seconds.".format(dt))

"""
Helper function to check the prediction for a single patch, used in
test_rf_feats_fullspec.

**This function is not updated for the HOG featureset that is now added by
default to the SliceInfo class. Therefore, you'll get an error and the
program will crash if you try to call this function right now.
"""
def check_classify(RF, kernels, patch, patch_label, i, tot):        
    # kernel the patch
    print("Checking Classifying patch {}/{}".format(i, tot))
    feat = compute_feats(patch, kernels)
    feat = feat.flatten().reshape(1, -1)
    prediction = RF.predict(feat)
    if prediction == patch_label:
        return True
    else:
        return False    