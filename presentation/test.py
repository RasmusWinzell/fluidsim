import applescript

# Save the group '#sim' in slide 1 as a PNG file
applescript.AppleScript(
    """tell application "Microsoft PowerPoint"
    activate
    
    set targetShapeName to "#sim"
    
    try
        -- Check if PowerPoint is running
        if not (running) then
            display dialog "Microsoft PowerPoint is not running. Please open it and try again."
            return
        end if
        
        -- Open the presentation if not already open
        set presentationName to "presentation" -- Replace with your actual presentation name
        try
            set targetPresentation to presentation presentationName
        on error
            set targetPresentation to open POSIX file "/Users/rasmuswinzell/Documents/TSBK03/fluidsim/presentation.pptx" -- Replace with the actual path to your presentation
        end try
        
        repeat with j from 1 to count of slides of targetPresentation
            set targetSlide to slide j of targetPresentation

            

            set shapeFound to false
            
            -- Iterate through each shape on the specified slide
            repeat with i from 1 to count of shapes of targetSlide
                set currentShape to shape i of targetSlide
                if name of currentShape is equal to targetShapeName then
                    set shapeFound to true
                    set imageFileName to "sim" & j & ".png" -- Replace with the actual path to your image
                    
                    -- Export the shape as an image
                    tell currentShape
                        save as picture file name imageFileName
                        set visible to false
                    end tell
                    exit repeat
                end if
            end repeat
        end repeat
    on error errMsg
        display dialog "Error: " & errMsg
    end try
end tell

"""
).run()
applescript.AppleScript(
    """tell application "Microsoft PowerPoint"
    activate
    
    set targetShapeName to "#sim"
    
    try
        -- Check if PowerPoint is running
        if not (running) then
            display dialog "Microsoft PowerPoint is not running. Please open it and try again."
            return
        end if
        
        -- Open the presentation if not already open
        set presentationName to "presentation" -- Replace with your actual presentation name
        try
            set targetPresentation to presentation presentationName
        on error
            set targetPresentation to open POSIX file "/Users/rasmuswinzell/Documents/TSBK03/fluidsim/presentation.pptx" -- Replace with the actual path to your presentation
        end try

        save targetPresentation in "slides.pdf" as save as PDF -- Replace with the actual path to your presentation
        
    on error errMsg
        display dialog "Error: " & errMsg
    end try
end tell

"""
).run()
applescript.AppleScript(
    """tell application "Microsoft PowerPoint"
    activate
    
    set targetShapeName to "#sim"
    
    try
        -- Check if PowerPoint is running
        if not (running) then
            display dialog "Microsoft PowerPoint is not running. Please open it and try again."
            return
        end if
        
        -- Open the presentation if not already open
        set presentationName to "presentation" -- Replace with your actual presentation name
        try
            set targetPresentation to presentation presentationName
        on error
            set targetPresentation to open POSIX file "/Users/rasmuswinzell/Documents/TSBK03/fluidsim/presentation.pptx" -- Replace with the actual path to your presentation
        end try
        
        repeat with j from 1 to count of slides of targetPresentation
            set targetSlide to slide j of targetPresentation

            

            set shapeFound to false
            
            -- Iterate through each shape on the specified slide
            repeat with i from 1 to count of shapes of targetSlide
                set currentShape to shape i of targetSlide
                if name of currentShape is equal to targetShapeName then
                    set shapeFound to true
                    set imageFileName to "sim" & j & ".png" -- Replace with the actual path to your image
                    
                    -- Export the shape as an image
                    tell currentShape
                        set visible to true
                    end tell
                    exit repeat
                end if
            end repeat
        end repeat
    on error errMsg
        display dialog "Error: " & errMsg
    end try
end tell
"""
).run()
import os
import re

from pdf2image import convert_from_path

save_dir = "/Users/rasmuswinzell/Library/Containers/com.microsoft.Powerpoint/Data/"
destination_dir = "/Users/rasmuswinzell/Documents/TSBK03/fluidsim/slide_images"
image_format = r"sim\d+.png"
pdf_fromat = r"slides.pdf"
slide_format = r"slide\d+.png"


# find all images
images = []
for file in os.listdir(save_dir):
    # print(file)
    if re.match(image_format, file):
        images.append(file)
        # Move file to destination
        os.rename(os.path.join(save_dir, file), os.path.join(destination_dir, file))

# find pdf
pdf = None
for file in os.listdir(save_dir):
    # print(file)
    if re.match(pdf_fromat, file):
        pdf = file
        # Put images in destination
        pages = convert_from_path(os.path.join(save_dir, file))
        for i, page in enumerate(pages):
            page.save(os.path.join(destination_dir, "slide{}.png".format(i)), "PNG")
