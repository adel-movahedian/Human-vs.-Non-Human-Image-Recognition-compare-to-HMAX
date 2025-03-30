function saveParticipantData(totalAcc, accuracyCloseBody, accuracyFarBody, accuracyHead, accuracyMediumBody, RT)
    currentTime = datetime('now', 'Format', 'yyyyMMdd_HHmmss');
    timestamp = datestr(currentTime, 'yyyymmdd_HHMMSS');
    subject = struct('TotalAccuracy', totalAcc, 'CloseAccuracy', accuracyCloseBody, 'FarAccuracy', accuracyFarBody, 'HeadAccuracy', accuracyHead, 'MediumAccuracy', accuracyMediumBody, 'AvgRT', mean(RT));
    save(['Subject', timestamp], 'subject');
    save(['RT', timestamp], 'RT');
end
