function img = loadAndDisplayImage(imagePath, window)
    img = rgb2gray(imread(imagePath));
    imageTexture = Screen('MakeTexture', window, img);
    Screen('DrawTexture', window, imageTexture);
    Screen('Flip', window);
    WaitSecs(0.02);
end