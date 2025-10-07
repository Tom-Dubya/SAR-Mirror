% Given:
% 1) the path to a directory containing files, OR
% 2) A path to a file listing paths to other files, 
% Iterate through said files and apply the function provided.
% The provided function must have signature matching the provided arguments
% with the first parameter being a string that encapsulates the absolute
% path to each file in question.
function IteratePath(dataPath, func, args)
    arguments
        dataPath (1,1) string
        func (1,1) function_handle
        args cell = {}
    end

    if exist(dataPath, 'file') == 2
        paths = readlines(dataPath);
    else
        paths = dir(fullfile(dataPath, '**', '*.*'));
        paths = paths(~[paths.isdir]);
        paths = fullfile({paths.folder}, {paths.name})';
    end

    count = numel(paths);
    for pathIndex = 1 : count
        currentPath = paths{pathIndex};
        if isempty(currentPath)
            continue;
        end

        disp("[DEBUG] Processing: " + currentPath + " (" + pathIndex + " of " + count + ")");
        func(currentPath, args{:});
    end
end