function [accuracyCloseBody, accuracyFarBody, accuracyHead, accuracyMediumBody, totalAcc] = calculateAccuracyPercentages(closeBodyNumber, farBodyNumber, headNumber, mediumBodyNumber, accuracyCloseBody, accuracyFarBody, accuracyHead, accuracyMediumBody)
    accuracyCloseBody = 100 * accuracyCloseBody / closeBodyNumber;
    accuracyFarBody = 100 * accuracyFarBody / farBodyNumber;
    accuracyHead = 100 * accuracyHead / headNumber;
    accuracyMediumBody = 100 * accuracyMediumBody / mediumBodyNumber;
    totalAcc = (accuracyCloseBody + accuracyFarBody + accuracyHead + accuracyMediumBody) / 4;

    disp(['Accuracy Close: ' num2str(accuracyCloseBody) '%']);
    disp(['Accuracy Far: ' num2str(accuracyFarBody) '%']);
    disp(['Accuracy Head: ' num2str(accuracyHead) '%']);
    disp(['Accuracy Medium: ' num2str(accuracyMediumBody) '%']);
    disp(['Total Accuracy: ' num2str(totalAcc) '%']);
end