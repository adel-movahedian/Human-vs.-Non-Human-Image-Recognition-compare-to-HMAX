%% Adel Movahedian & Pouya Farivar
%%
 clc; clear;
KbName('UnifyKeyNames');

% ---------- Set Parameters ----------
Hzs = 15; % Frequency of stimulus presentation
TrialsPerBlock = 120; % Number of trials per block
TrainingBlocks = 5; % Number of training blocks
TestingBlocks = 5; % Number of testing blocks
RT = 3; % Rest time (seconds) between blocks

% Psychtoolbox setup
Screen('Preference', 'SkipSyncTests', 1);
[WinPointer, WinSize] = Screen('OpenWindow', 0);
CenterX = WinSize(3) / 2; 
CenterY = WinSize(4) / 2;

% Screen properties
WhiteColor = [255 255 255];
GrayColor = [128 128 128];
BackColor = [15 15 15];

% Dataset paths
trainAnimalPath = 'Dataset/Train/Train_Animals';
trainNonAnimalPath = 'Dataset/Train/Train_Non-Animals';
testAnimalPath = 'Dataset/Test/Test_Animals';
testNonAnimalPath = 'Dataset/Test/Test_Non-Animals';

% Load datasets
trainingImages = LoadDataset(trainAnimalPath, trainNonAnimalPath, '*.jpg');
testingImages = LoadDataset(testAnimalPath, testNonAnimalPath, '*.jpg');

% Check dataset loading
if isempty(trainingImages)
    error('Training images not loaded. Check dataset structure.');
end
if isempty(testingImages)
    error('Testing images not loaded. Check dataset structure.');
end

% Initialize results
Accuracy = [];
ResponseTime = [];
Confidence = [];

% % ---------- Training Phase ----------
% for block = 1:TrainingBlocks
%     fprintf('Training Block %d\n', block);
%     [acc, rt, conf] = RunBlock(WinPointer, trainingImages, TrialsPerBlock, GrayColor, WhiteColor, BackColor);
%     Accuracy = [Accuracy; acc];
%     ResponseTime = [ResponseTime; rt];
%     Confidence = [Confidence; conf];
%     RestPeriod(WinPointer, BackColor, WhiteColor, RT);
% end

% ---------- Testing Phase ----------
for block = 1:TestingBlocks
    fprintf('Testing Block %d\n', block);
    % Apply transformations in the last two blocks
    applyTransformations = (block > TestingBlocks - 2);
    [acc, rt, conf] = RunBlock(WinPointer, testingImages, TrialsPerBlock, GrayColor, WhiteColor, BackColor, applyTransformations);
    Accuracy = [Accuracy; acc];
    ResponseTime = [ResponseTime; rt];
    Confidence = [Confidence; conf];
    RestPeriod(WinPointer, BackColor, WhiteColor, RT);
end

% Save results
save('TaskResults_subjectNumber1.mat', 'Accuracy', 'ResponseTime', 'Confidence');

% Close the window
Screen('CloseAll');

%% --- Custom Functions ---

function images = LoadDataset(animalPath, nonAnimalPath, fileFormat)
    % Load images from the specified paths
    images = [];
    
    % Define category mapping based on first character in filename
    categoryMap = struct('H', 'Head', 'N', 'Near-Body', 'M', 'Middle-Body', 'F', 'Far-Body');
    
    % Load animal images
    animalFiles = dir(fullfile(animalPath, fileFormat));
    for i = 1:length(animalFiles)
        filename = animalFiles(i).name;
        category = GetCategoryFromFilename(filename, categoryMap);
        img = imread(fullfile(animalPath, filename));
        img = imresize(img, [256, 256]); % Resize for consistency
        images = [images; {img, 1, category}]; % Label: 1 = Animal
    end
    
    % Load non-animal images
    nonAnimalFiles = dir(fullfile(nonAnimalPath, fileFormat));
    for i = 1:length(nonAnimalFiles)
        filename = nonAnimalFiles(i).name;
        category = GetCategoryFromFilename(filename, categoryMap);
        img = imread(fullfile(nonAnimalPath, filename));
        img = imresize(img, [256, 256]); % Resize for consistency
        images = [images; {img, 0, category}]; % Label: 0 = Non-Animal
    end
end

function category = GetCategoryFromFilename(filename, categoryMap)
    % Extract the first character from the filename
    firstChar = upper(filename(1));
    if isfield(categoryMap, firstChar)
        category = categoryMap.(firstChar);
    else
        error('Unknown category in filename: %s', filename);
    end
end

function [accuracy, responseTime, confidence] = RunBlock(WinPointer, images, numTrials, GrayColor, WhiteColor, BackColor, applyTransformations)
    % Run a block of trials with the given images
    accuracy = 0;
    responseTimes = zeros(1, numTrials);
    confidences = zeros(1, numTrials);
    
    % Screen dimensions for confidence bar
    [screenX, screenY] = Screen('WindowSize', WinPointer);
    barWidth = 50; % Width of the confidence bar
    barHeight = screenY; % Full screen height
    
    for trial = 1:numTrials
        % Select a random image
        imgIdx = randi(size(images, 1));
        stimulus = images{imgIdx, 1};
        isAnimal = images{imgIdx, 2};
        category = images{imgIdx, 3}; % Category: Head, Near-Body, etc.

        if applyTransformations
            if rand < 0.5
                stimulus = AddNoise(stimulus); % Add noise with 50% probability
            else
                stimulus = RotateImage(stimulus); % Rotate with 50% probability
            end
        end
        % 1. Fixation cross
        Screen('FillRect', WinPointer, BackColor);
        DrawFormattedText(WinPointer, '+', 'center', 'center', WhiteColor);
        Screen('Flip', WinPointer);
        WaitSecs(0.5); % 500 ms
        
        % 2. Stimulus presentation
        Screen('PutImage', WinPointer, stimulus);
        Screen('Flip', WinPointer);
        WaitSecs(0.02); % 20 ms
        
        % 3. ISI (Inter-stimulus Interval)
        Screen('FillRect', WinPointer, GrayColor);
        Screen('Flip', WinPointer);
        WaitSecs(0.03); % 30 ms
        
        % 4. Mask presentation
        mask = ScrambleStimulus(stimulus); % Create scrambled mask
        Screen('PutImage', WinPointer, mask);
        Screen('Flip', WinPointer);
        WaitSecs(0.08); % 80 ms
        
        % 5. Decision and confidence rating
        responseRecorded = false; % Track if the response has been recorded
        trialStart = GetSecs(); % Start timing the trial
        
        while ~responseRecorded
            % Get mouse position and button state
            [x, y, buttons] = GetMouse(WinPointer);
            KbName('UnifyKeyNames'); % Ensure unified key names
            escapeKey = KbName('ESCAPE');
            [keyIsDown, ~, keyCode] = KbCheck;
            if keyIsDown && keyCode(escapeKey)
                Screen('CloseAll');
                disp('Task completed or interrupted. Resources released.');
            end
            % Draw the decision prompt
            Screen('FillRect', WinPointer, BackColor);
            DrawFormattedText(WinPointer, 'Left Click: Non-Animal | Right Click: Animal', 'center', screenY * 0.1, WhiteColor);
            
            % Draw the confidence bar
            confidenceLevel = 1 - (y / screenY); % Map vertical position to confidence (0 to 1)
            confidenceLevel = max(0, min(1, confidenceLevel)); % Clamp between 0 and 1
            barColor = [0, 255 * confidenceLevel, 255 * (1 - confidenceLevel)]; % Gradient from blue (low) to green (high)
            barRect = [screenX - barWidth, 0, screenX, screenY]; % Position the bar on the right edge
            fillHeight = confidenceLevel * barHeight; % Height of the filled portion
            
            % Draw the confidence bar (dynamic portion)
            Screen('FillRect', WinPointer, [50, 50, 50], barRect); % Background of the bar
            Screen('FillRect', WinPointer, barColor, [barRect(1), barRect(4) - fillHeight, barRect(3), barRect(4)]); % Filled portion
            
            Screen('Flip', WinPointer); % Update the screen
            
            % Check for mouse clicks
            if buttons(1) % Left mouse button
                userResponse = 0; % Non-Animal
                responseRecorded = true;
            elseif buttons(3) % Right mouse button
                userResponse = 1; % Animal
                responseRecorded = true;
            end
        end
        
        responseTimes(trial) = GetSecs() - trialStart; % Record response time
        confidences(trial) = confidenceLevel; % Save confidence
        
        % Check if the response is correct
        if userResponse == isAnimal
            accuracy = accuracy + 1;
        end
    end
    
    % Compute metrics
    accuracy = accuracy / numTrials; % Proportion correct
    responseTime = mean(responseTimes);
    confidence = mean(confidences);
end



function scrambled = ScrambleStimulus(image)
    % Scramble the pixels of an image to create a mask
    [rows, cols, ~] = size(image);
    scrambled = image(randperm(rows), randperm(cols), :);
end

function RestPeriod(WinPointer, BackColor, WhiteColor, RT)
    % Display rest screen
    Screen('FillRect', WinPointer, BackColor);
    DrawFormattedText(WinPointer, 'Rest! Press any key to continue', 'center', 'center', WhiteColor);
    Screen('Flip', WinPointer);
    
    fprintf('Press any key to continue...\n');
    GetChar; % Wait for any key press
    WaitSecs(RT); % Rest time
end

function noisyImage = AddNoise(image)
    % Add Gaussian noise to the image
    noise = uint8(randn(size(image)) * 20); % Scale noise intensity
    noisyImage = imadd(image, noise);
end

function rotatedImage = RotateImage(image)
    % Rotate the image by a random angle
    angle = randi([-30, 30]); % Random angle between -30 and 30 degrees
    rotatedImage = imrotate(image, angle, 'crop');
end
