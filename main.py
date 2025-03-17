# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors
from shapely.geometry import Polygon, Point, LineString  # ✅ Added missing imports

class ObjectCounter(BaseSolution):
    """
    A class to manage the counting of objects in a real-time video stream based on their tracks.
    """

    def __init__(self, **kwargs):
        """Initializes the ObjectCounter class for real-time object counting in video streams."""
        super().__init__(**kwargs)

        self.in_count = 0  
        self.out_count = 0  
        self.counted_ids = []  
        self.classwise_counts = {}  
        self.track_history = {}  # ✅ Initialized missing attribute

        self.region = []  # ✅ Ensure region is initialized
        self.region_initialized = False  

        self.show_in = self.CFG.get("show_in", True)
        self.show_out = self.CFG.get("show_out", True)

    def count_objects(self, current_centroid, track_id, prev_position, cls):
        """Counts objects within a polygonal or linear region based on their tracks."""
        if prev_position is None or track_id in self.counted_ids:
            return
        
        if not self.region:  # ✅ Ensure region is not empty
            return

        if len(self.region) == 2:  # Linear region (line segment)
            line = LineString(self.region)
            if line.intersects(LineString([prev_position, current_centroid])):
                if abs(self.region[0][0] - self.region[1][0]) < abs(self.region[0][1] - self.region[1][1]):
                    if current_centroid[0] > prev_position[0]:  
                        self.in_count += 1
                        self.classwise_counts[self.names.get(cls, "Unknown")]["IN"] += 1  
                    else:
                        self.out_count += 1
                        self.classwise_counts[self.names.get(cls, "Unknown")]["OUT"] += 1
                elif current_centroid[1] > prev_position[1]:  
                    self.in_count += 1
                    self.classwise_counts[self.names.get(cls, "Unknown")]["IN"] += 1
                else:  
                    self.out_count += 1
                    self.classwise_counts[self.names.get(cls, "Unknown")]["OUT"] += 1
                self.counted_ids.append(track_id)

        elif len(self.region) > 2:  # Polygonal region
            polygon = Polygon(self.region)
            if polygon.contains(Point(current_centroid)):
                region_width = max(p[0] for p in self.region) - min(p[0] for p in self.region)
                region_height = max(p[1] for p in self.region) - min(p[1] for p in self.region)

                if (
                    (region_width < region_height and current_centroid[0] > prev_position[0])
                    or (region_width >= region_height and current_centroid[1] > prev_position[1])
                ):  
                    self.in_count += 1
                    self.classwise_counts[self.names.get(cls, "Unknown")]["IN"] += 1
                else:  
                    self.out_count += 1
                    self.classwise_counts[self.names.get(cls, "Unknown")]["OUT"] += 1
                self.counted_ids.append(track_id)

    def store_classwise_counts(self, cls):
        """Initialize class-wise counts if not already present."""
        class_name = self.names.get(cls, "Unknown")  
        if class_name not in self.classwise_counts:
            self.classwise_counts[class_name] = {"IN": 0, "OUT": 0}

    def display_counts(self, plot_im):
        """Display object counts on the input image."""
        labels_dict = {
            str.capitalize(key): f"{'IN ' + str(value['IN']) if self.show_in else ''} "
            f"{'OUT ' + str(value['OUT']) if self.show_out else ''}".strip()
            for key, value in self.classwise_counts.items()
            if value["IN"] != 0 or value["OUT"] != 0
        }
        if labels_dict:
            self.annotator.display_analytics(plot_im, labels_dict, (104, 31, 17), (255, 255, 255), 10)

    def process(self, im0):
        """Process input frames and update object counts."""
        if not self.region_initialized:
            self.initialize_region()
            self.region_initialized = True

        self.extract_tracks(im0)  
        self.annotator = SolutionAnnotator(im0, line_width=self.line_width)  

        self.annotator.draw_region(
            reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2
        )  

        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            self.annotator.box_label(box, label=self.names.get(cls, "Unknown"), color=colors(cls, True))
            self.track_history.setdefault(track_id, []).append(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
            self.store_classwise_counts(cls)

            prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None
            self.count_objects(self.track_history[track_id][-1], track_id, prev_position, cls)

        plot_im = self.annotator.result()
        self.display_counts(plot_im)  
        self.display_output(plot_im)  

        return SolutionResults(
            plot_im=plot_im,
            in_count=self.in_count,
            out_count=self.out_count,
            classwise_count=self.classwise_counts,
            total_tracks=len(self.track_ids),
        )
