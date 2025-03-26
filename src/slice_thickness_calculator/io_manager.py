import sys
import json
from pathlib import Path
import warnings

import time
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)


class IO:
    if getattr(sys, "frozen", False):
        project_dir = Path(sys.executable).parent
    else:
        project_dir = Path(__file__).parent.parent.parent

    with open(project_dir.joinpath("config/file_structure.json")) as f:
        config = json.load(f)

    paths_from_proj_dir = {
        "image_input_dir": project_dir.joinpath(config["image_input_dir"]),
        "figures_dir": project_dir.joinpath(config["figures_dir"]),
        "excel_log": project_dir.joinpath(config["excel_log"]),
    }

    log_template = pd.DataFrame(
        columns=["File Name", "Processing Timestamp"]
        + [f"Slice {i}" for i in range(1, 15)]
    )

    @classmethod
    def creation_control(cls):
        creation_not_required = True
        for path in cls.paths_from_proj_dir.values():
            if not path.exists():
                creation_not_required = False
                if path.suffix == ".xlsx":
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch(exist_ok=True)
                    cls.log_template.to_excel(path, index=False)
                    print(f"Excel file created: {path.absolute()}")
                else:
                    path.mkdir(parents=True, exist_ok=True)
                    print(f"Directory created: {path.absolute()}")

        if creation_not_required:
            print("All I/O paths exist, as expected!")

        else:
            print(
                "Some I/O paths did not exist!\n"
                "Issue rectified. Please re-run the software."
            )
            time.sleep(10)
            sys.exit()

    @classmethod
    def pull_images(cls):
        file_types = [".jpg", ".jpeg"]
        already_analysed = cls.pull_already_analysed_images()
        images_to_analyse = [
            img
            for f in file_types
            for img in cls.paths_from_proj_dir["image_input_dir"].rglob(f"*{f}")
            if img.name not in already_analysed
        ]
        if images_to_analyse:
            return images_to_analyse
        else:
            print(
                f"Error: all images in image input folder have already been analysed!"
            )
            time.sleep(5)
            sys.exit()

    @classmethod
    def pull_already_analysed_images(cls):
        df = pd.read_excel(cls.paths_from_proj_dir["excel_log"])
        analysed_images = df[df.columns[0]].to_list()
        return analysed_images

    @classmethod
    def update_log(cls, image: "SliceThicknessImage"):
        df = pd.read_excel(cls.paths_from_proj_dir["excel_log"])

        # get new row
        new_row = (
            [image.path.name, pd.Timestamp.now()]
            + [round(peak.FWHM * image.mm_cf, 1) for peak in image.line_profile.peaks]
            + [None] * (15 - len(image.line_profile.peaks))
        )

        new_row_df = pd.DataFrame([dict(zip(df.columns, new_row))])

        # contact new row to df or create new df if empty
        if not (df.empty or df.isna().all().all()):
            df = pd.concat(
                [df.dropna(axis=0, how="all"), new_row_df], ignore_index=True, axis=0
            )
        else:
            df = new_row_df

        with pd.ExcelWriter(
            cls.paths_from_proj_dir["excel_log"], engine="xlsxwriter"
        ) as writer:
            df.to_excel(writer, index=False)

            workbook = writer.book
            worksheet = writer.sheets.values().__iter__().__next__()
            worksheet.set_column(0, 0, 30)
            worksheet.set_column(1, 1, 20)
            worksheet.set_column(2, 16, 10)
            worksheet.freeze_panes(1, 0)
