import qualipy as qp
import qualipy.filters as qpf
import json
import glob
from joblib import Parallel, delayed

def compute_qualipy_stats(image):
    # print("Processing {} ...".format(image))
    results = qp.process(
        image,
        [
            qpf.WholeBlur(),
            qpf.BlurredContext(),
            qpf.Exposure(negative_under_exposed=True),
            # qpf.Highlights(),
            # qpf.CrossProcessed(),
        ],
        combine_results=False,
        return_predictions=True
    )
    return results


threads = False
n_jobs = -2
# paths = glob.glob("/opt/app/image-quality-assessment/test_images/*.jpg")
with open("/opt/app/image-quality-assessment/qualipy_images.txt", "r") as f:
    paths = [p.strip() for p in f.readlines()]
print(paths)

if threads:
    par = Parallel(n_jobs=n_jobs, prefer="threads", verbose=10)
else:
    par = Parallel(n_jobs=n_jobs, prefer="processes", verbose=10)
results = par(
    delayed(compute_qualipy_stats)(path) for path in paths
)
# print(results)
fname = "/opt/app/image-quality-assessment/qualipy_analysis_results.json"
with open(fname, "w") as f:
    json.dump(results, f)
# print("File written")
# with open(fname, 'r') as f:
#     print(f.read())
# print(results)
