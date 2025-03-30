function [subjectAnswer, RT] = collectResponse(window, kSpace, kQ, subjectAnswer, trueAnswer, i, group)
    keyIsDown = 0;
    flag = 0;
    startTime = GetSecs;
    subjectAnswer(i) = -1;

    while ~flag
        [keyIsDown, ~, keyCode] = KbCheck();

        if keyIsDown
            key = KbName(keyCode);
            hh = strcmp(key, 'a');
            gg = strcmp(key, 'n');

            stopTime = GetSecs();
            RT(i) = stopTime - startTime;

            if (strcmp(key(1), 'a') || hh(1))
                subjectAnswer(i) = 0;
                flag = 1;
            elseif (strcmp(key(1), 'n') || gg(1))
                subjectAnswer(i) = 1;
                flag = 1;
            end
        end
    end
end