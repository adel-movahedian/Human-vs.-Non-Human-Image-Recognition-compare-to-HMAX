function [group, closeBodyNumber, farBodyNumber, headNumber, mediumBodyNumber] = determineCategory(imageName, closeBodyNumber, farBodyNumber, headNumber, mediumBodyNumber)
    group = nan;

    if imageName(48) == 'B'
        group = 'Close';
        closeBodyNumber = closeBodyNumber + 1;
    elseif imageName(48) == 'F'
        group = 'Far';
        farBodyNumber = farBodyNumber + 1;
    elseif imageName(48) == 'H'
        group = 'Head';
        headNumber = headNumber + 1;
    elseif imageName(48) == 'M'
        group = 'Medium';
        mediumBodyNumber = mediumBodyNumber + 1;
    end
end