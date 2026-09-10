# Documentation
---


## Markup
`markup.py` is a tool for manually annotating iris images.

You can use it to annotate the **pupil** and **iris** by selecting **3 points for each circle**. The script calculates the corresponding circle center and radius automatically from these points.

The tool also supports eyelid annotation.

### Usage

```bash
> python markup.py --mode circles --input-dir ./images --output-dir ./output
```

[Example of usage](image_markup_iris.jpg)


### Arguments

- **`--mode {circles,eyelids}`**  
  Selects the annotation mode.

  - `circles` — annotate the pupil and iris using 3 points for each circle.
  - `eyelids` — annotate the upper and lower eyelids using 3 points for each eyelid.

- **`--input-dir INPUT_DIR`**  
  Path to the directory containing the images that should be annotated.

- **`--output-dir OUTPUT_DIR`**  
  Path to the directory where generated outputs, normalized images, masks, or other annotation results will be saved.

- **`--circle-model CIRCLE_MODEL`**  
  Path to the trained YOLO model weights used for automatic pupil and iris detection.

  This option is particularly useful in `eyelids` mode. The detected pupil and iris circles are used as the geometric basis for eyelid annotation and iris normalization. The eyelid mask can then be applied to the normalized iris image to remove occluded regions.

- **`--log-file LOG_FILE`**  
  Path to the CSV file where annotation results and metadata will be stored.

  Default log files:

  - `annotations_log.csv` — for `circles` mode
  - `annotations_log_2.csv` — for `eyelids` mode

### Circle Annotation

In `circles` mode, annotate the image in the following order:

1. Select 3 points on the pupil boundary.
2. Select 3 points on the iris boundary.

The script calculates:

- pupil center `(cx, cy)`
- pupil radius
- iris center `(cx, cy)`
- iris radius

These values can later be used for iris normalization or for creating a training dataset for object detection models.

### Eyelid Annotation

In `eyelids` mode, annotate:

1. 3 points on the upper eyelid
2. 3 points on the lower eyelid

The points are used to approximate the eyelid curves and generate an eyelid mask.

The mask can then be normalized together with the iris image and used to remove regions occluded by the eyelids.

