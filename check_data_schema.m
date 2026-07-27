%% Validate the private response-matrix input without starting training

inputFile = 'Response_Matrix_Table.csv';
requiredColumns = [ ...
    "case_id", "sensor_id", "x_cm", "y_cm", "layer", "direction", ...
    "delta_lambda_resp_nm", "load_type", "load_kg", ...
    "p1_x", "p1_y", "p2_x", "p2_y" ...
];

assert(isfile(inputFile), ...
    'Missing private input file: %s', inputFile);

T = readtable(inputFile, 'TextType', 'string');
actualColumns = string(T.Properties.VariableNames);
missingColumns = setdiff(requiredColumns, actualColumns);

assert(isempty(missingColumns), ...
    'Input table is missing required columns: %s', ...
    strjoin(missingColumns, ', '));

assert(height(T) > 0, 'The input table contains no rows.');

fprintf('Schema check passed.\n');
fprintf('Rows: %d\n', height(T));
fprintf('Columns: %d\n', width(T));
fprintf('Cases: %d\n', numel(unique(string(T.case_id))));
fprintf('Layers: %s\n', strjoin(unique(string(T.layer)), ', '));
fprintf('Directions: %s\n', strjoin(unique(string(T.direction)), ', '));
