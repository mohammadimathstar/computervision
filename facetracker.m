close all
clear all

expr = '(?<seqID>\d+)_(?<subID>\d+)_(?<segID>\d+)_(?<name>\w+).avi';
% expr_f = '(?<idx>\d+)_(?<name>\w+)_(?<subID>\d+)_(?<segID>\d+).mat';

initfolder = "../ytcelebrity_init/Init_State/";
list_videos = dir("*.avi");

t=1;
for i=1:length(list_videos)
    i
    videoname = list_videos(i).name;
    a = regexp(videoname, expr, 'names');
    init_file = strcat(initfolder, "*_", a.name, "_", a.subID, "_", a.segID, ".mat"); 
    load( strcat(initfolder, dir(init_file).name) );
    bbox = computebBox(gp);
    
    parfolder = strcat("./test/",a.name, "/");
%     mkdir(parfolder)
    chifolder = strcat(parfolder, num2str(a.seqID), "_", num2str(a.subID), "_", num2str(a.segID), "/");
    mkdir(chifolder)
%     if t>0 && t<5
        images=tracker(videoname, bbox, 30);
        address = chifolder;
        saveimages(images, address)
%         drawbox(videoname, bbox)        
%     end
%     t=t+1;
end


function bbox = computebBox(gp)
    xc = gp(1); yc = gp(2);
    q = gp(4);
    s = gp(3);
    width = s*48 ; height = s*48 ;
    x = xc - width / 2; y = yc - height / 2;

    bbox = [x, y, width, height];
end

function drawbox(videoname, bbox)
    % Read a video frame and run the face detector.
    videoReader = VideoReader(videoname);
    videoFrame      = readFrame(videoReader);
    
    % Draw the returned bounding box around the detected face.
    videoFrame = insertShape(videoFrame, 'Rectangle', bbox);
    bboxPoints = bbox2points(bbox(1, :));
    points = detectMinEigenFeatures(rgb2gray(videoFrame), 'ROI', bbox);

    % Display the detected points.
    figure, imshow(videoFrame), hold on, title('Detected features');
    plot(points);

%     figure; imshow(videoFrame); title('Detected face');
end

function allframes = tracker(videoname, bbox, outsize)
    % Read a video frame and run the face detector.
    videoReader = VideoReader(videoname);
    videoFrame      = readFrame(videoReader);
    nframe = videoReader.Duration * videoReader.FrameRate - 1;
%     allframes = zeros( outsize, outsize, nframe);
  
    
    % Draw the returned bounding box around the detected face.
%     videoFrame = insertShape(videoFrame, 'Rectangle', bbox);
    bboxPoints = bbox2points(bbox(1, :));
    points = detectMinEigenFeatures(rgb2gray(videoFrame), 'ROI', bbox);
    
    % tracker
    pointTracker = vision.PointTracker('MaxBidirectionalError', 2);

    % Initialize the tracker with the initial point locations and the initial
    % video frame.
    points = points.Location;
    initialize(pointTracker, points, videoFrame);

    % initialize video player
    videoPlayer  = vision.VideoPlayer('Position',...
        [100 100 [size(videoFrame, 2), size(videoFrame, 1)]+30]);
    t=1;
%     nempty=0;
    oldPoints = points;
    while hasFrame(videoReader)
        % get the next frame
        videoFrame = readFrame(videoReader);
        

        % Track the points. Note that some points may be lost.
        [points, isFound] = step(pointTracker, videoFrame);
        visiblePoints = points(isFound, :);
        oldInliers = oldPoints(isFound, :);

        if size(visiblePoints, 1) >= 2 % need at least 2 points

            % Estimate the geometric transformation between the old points
            % and the new points and eliminate outliers
            [xform, inlierIdx] = estimateGeometricTransform2D(...
                oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);
%             oldInliers    = oldInliers(inlierIdx, :);
            visiblePoints = visiblePoints(inlierIdx, :);

            % Apply the transformation to the bounding box points
            bboxPoints = transformPointsForward(xform, bboxPoints);
            boxPolygon = bboxPoints;         
            
            m = floor(min(boxPolygon));
            M = floor(max(boxPolygon));
            if M(1)>size(videoFrame, 2)
                M(1)=size(videoFrame,2);
            end
            if M(2)>size(videoFrame,1)
                M(2)=size(videoFrame,1);
            end
            if m(1)<1
                m(1)=1;
            end
            if m(2)<1
                m(2)=1;
            end
            idx = m(2):M(2);
            idy = m(1):M(1);
            
            cropped = rgb2gray(videoFrame(idx, idy,:));
            allframes(:,:, t) = imresize(cropped,[outsize outsize]);
            imshow(allframes(:,:, t))
            
            
            % Reset the points
            oldPoints = visiblePoints;
            setPoints(pointTracker, oldPoints);  
            t = t+1;
        
        end
        
    end
    
    % Clean up
    release(videoPlayer);
end

function saveimages(images, address)
    for i=1:size(images,3)
        imwrite(images(:,:,i), strcat(address, num2str(i), ".png"));
        
    end
end