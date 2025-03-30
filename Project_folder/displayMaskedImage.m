function displayMaskedImage(window, img, waitTimeBeforeFlip, maskedDisplayTime)
    Screen('Flip', window);
    WaitSecs(waitTimeBeforeFlip);
    vec = randperm(numel(img));
    vec = reshape(vec, size(img));
    out = img(vec);
    maskTexture = Screen('MakeTexture', window, out);
    Screen('DrawTexture', window, maskTexture);
    Screen('Flip', window);
    WaitSecs(maskedDisplayTime);
end