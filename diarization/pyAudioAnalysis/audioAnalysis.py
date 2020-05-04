from __future__ import print_function
import argparse
import audioSegmentation as aS
import analysis_results

def speakerDiarizationWrapper(inputFile, numSpeakers):
    timeStamps, sampling_rate, num_frames = aS.speaker_diarization(inputFile, numSpeakers, lda_dim=0, plot_res=True)
    duration = num_frames/float(sampling_rate)
    analysis_results.preprocess_diarization_results(duration,timeStamps)

def parse_arguments():
    parser = argparse.ArgumentParser(description="A demonstration script "
                                                 "for pyAudioAnalysis library")
    tasks = parser.add_subparsers(
        title="subcommands", description="available tasks",
        dest="task", metavar="")

    dirMp3Wav = tasks.add_parser("dirMp3toWav",
                                 help="Convert all .mp3 files in a directory "
                                      "to .wav format")
    dirMp3Wav.add_argument("-i", "--input", required=True, help="Input folder")
    dirMp3Wav.add_argument("-r", "--rate", type=int,
                           choices=[8000, 16000, 32000, 44100], required=True,
                           help="Samplerate of generated WAV files")
    dirMp3Wav.add_argument("-c", "--channels", type=int, choices=[1, 2],
                           required=True,
                           help="Audio channels of generated WAV files")

    dirWavRes = tasks.add_parser("dirWavResample",
                                 help="Change samplerate of .wav "
                                      "files in a directory")
    dirWavRes.add_argument("-i", "--input", required=True, help="Input folder")
    dirWavRes.add_argument("-r", "--rate", type=int,
                           choices=[8000, 16000, 32000, 44100], required=True,
                           help="Samplerate of generated WAV files")
    dirWavRes.add_argument("-c", "--channels", type=int, choices=[1, 2],
                           required=True,
                           help="Audio channels of generated WAV files")

    featExt = tasks.add_parser("featureExtractionFile",
                               help="Extract audio features from file")
    featExt.add_argument("-i", "--input", required=True,
                         help="Input audio file")
    featExt.add_argument("-o", "--output", required=True,
                         help="Output file")
    featExt.add_argument("-mw", "--mtwin", type=float,
                         required=True, help="Mid-term window size")
    featExt.add_argument("-ms", "--mtstep", type=float,
                         required=True, help="Mid-term window step")
    featExt.add_argument("-sw", "--stwin", type=float,
                         default=0.050, help="Short-term window size")
    featExt.add_argument("-ss", "--ststep", type=float,
                         default=0.050, help="Short-term window step")

    beat = tasks.add_parser("beatExtraction",
                            help="Compute beat features of an audio file")
    beat.add_argument("-i", "--input", required=True, help="Input audio file")
    beat.add_argument("--plot", action="store_true", help="Generate plot")

    featExtDir = tasks.add_parser("featureExtractionDir",
                                  help="Extract audio features "
                                       "from files in a folder")
    featExtDir.add_argument("-i", "--input", required=True,
                            help="Input directory")
    featExtDir.add_argument("-mw", "--mtwin", type=float, required=True,
                            help="Mid-term window size")
    featExtDir.add_argument("-ms", "--mtstep", type=float, required=True,
                            help="Mid-term window step")
    featExtDir.add_argument("-sw", "--stwin", type=float, default=0.050,
                            help="Short-term window size")
    featExtDir.add_argument("-ss", "--ststep", type=float, default=0.050,
                            help="Short-term window step")

    featVis = tasks.add_parser("featureVisualization")
    featVis.add_argument("-i", "--input", required=True, help="Input directory")

    spectro = tasks.add_parser("fileSpectrogram")
    spectro.add_argument("-i", "--input", required=True,
                         help="Input audio file")

    chroma = tasks.add_parser("fileChromagram")
    chroma.add_argument("-i", "--input", required=True, help="Input audio file")

    trainClass = tasks.add_parser("trainClassifier",
                                  help="Train an SVM or KNN classifier")
    trainClass.add_argument("-i", "--input", nargs="+",
                            required=True, help="Input directories")
    trainClass.add_argument("--method",
                            choices=["svm", "svm_rbf", "knn", "randomforest",
                                     "gradientboosting","extratrees"],
                            required=True, help="Classifier type")
    trainClass.add_argument("--beat", action="store_true",
                            help="Compute beat features")
    trainClass.add_argument("-o", "--output", required=True,
                            help="Generated classifier filename")

    trainReg = tasks.add_parser("trainRegression")
    trainReg.add_argument("-i", "--input", required=True,
                          help="Input directory")
    trainReg.add_argument("--method", choices=["svm", "randomforest","svm_rbf"],
                          required=True, help="Classifier type")
    trainReg.add_argument("--beat", action="store_true",
                          help="Compute beat features")
    trainReg.add_argument("-o", "--output", required=True,
                          help="Generated classifier filename")

    classFile = tasks.add_parser("classifyFile",
                                 help="Classify a file using an "
                                      "existing classifier")
    classFile.add_argument("-i", "--input", required=True,
                           help="Input audio file")
    classFile.add_argument("--model", choices=["svm", "svm_rbf", "knn",
                                               "randomforest",
                                               "gradientboosting",
                                               "extratrees"],
                           required=True, help="Classifier type (svm or knn or"
                                               " randomforest or "
                                               "gradientboosting or "
                                               "extratrees)")
    classFile.add_argument("--classifier", required=True,
                           help="Classifier to use (path)")

    trainHMM = tasks.add_parser("trainHMMsegmenter_fromfile",
                                help="Train an HMM from file + annotation data")
    trainHMM.add_argument("-i", "--input", required=True,
                          help="Input audio file")
    trainHMM.add_argument("--ground", required=True,
                          help="Ground truth path (segments CSV file)")
    trainHMM.add_argument("-o", "--output", required=True,
                          help="HMM model name (path)")
    trainHMM.add_argument("-mw", "--mtwin", type=float, required=True,
                          help="Mid-term window size")
    trainHMM.add_argument("-ms", "--mtstep", type=float, required=True,
                          help="Mid-term window step")

    trainHMMDir = tasks.add_parser("trainHMMsegmenter_fromdir",
                                   help="Train an HMM from file + annotation "
                                        "data stored in a directory (batch)")
    trainHMMDir.add_argument("-i", "--input", required=True,
                             help="Input audio folder")
    trainHMMDir.add_argument("-o", "--output", required=True,
                             help="HMM model name (path)")
    trainHMMDir.add_argument("-mw", "--mtwin", type=float, required=True,
                             help="Mid-term window size")
    trainHMMDir.add_argument("-ms", "--mtstep", type=float, required=True,
                             help="Mid-term window step")

    segmentClassifyFile = tasks.add_parser("segmentClassifyFile",
                                           help="Segmentation - classification "
                                                "of a WAV file given a trained "
                                                "SVM or kNN")
    segmentClassifyFile.add_argument("-i", "--input", required=True,
                                     help="Input audio file")
    segmentClassifyFile.add_argument("--model",
                                     choices=["svm", "svm_rbf", "knn",
                                              "randomforest","gradientboosting",
                                              "extratrees"],
                                     required=True, help="Model type")
    segmentClassifyFile.add_argument("--modelName", required=True,
                                     help="Model path")

    segmentClassifyFileHMM = tasks.add_parser("segmentClassifyFileHMM",
                                              help="Segmentation - "
                                                   "classification of a WAV "
                                                   "file given a trained HMM")
    segmentClassifyFileHMM.add_argument("-i", "--input", required=True,
                                        help="Input audio file")
    segmentClassifyFileHMM.add_argument("--hmm", required=True,
                                        help="HMM Model to use (path)")

    segmentationEvaluation = tasks.add_parser("segmentationEvaluation", help=
                                              "Segmentation - classification "
                                              "evaluation for a list of WAV "
                                              "files and CSV ground-truth "
                                              "stored in a folder")
    segmentationEvaluation.add_argument("-i", "--input", required=True,
                                        help="Input audio folder")
    segmentationEvaluation.add_argument("--model",
                                        choices=["svm", "knn", "hmm"],
                                        required=True, help="Model type")
    segmentationEvaluation.add_argument("--modelName", required=True,
                                        help="Model path")

    regFile = tasks.add_parser("regressionFile")
    regFile.add_argument("-i", "--input", required=True,
                         help="Input audio file")
    regFile.add_argument("--model", choices=["svm", "svm_rbf","randomforest"],
                         required=True, help="Regression type")
    regFile.add_argument("--regression", required=True,
                         help="Regression model to use")

    classFolder = tasks.add_parser("classifyFolder")
    classFolder.add_argument("-i", "--input", required=True,
                             help="Input folder")
    classFolder.add_argument("--model", choices=["svm", "svm_rbf", "knn",
                                                 "randomforest",
                                                 "gradientboosting",
                                                 "extratrees"],
                             required=True, help="Classifier type")
    classFolder.add_argument("--classifier", required=True,
                             help="Classifier to use (filename)")
    classFolder.add_argument("--details", action="store_true",
                             help="Plot details (otherwise only "
                                  "counts per class are shown)")

    regFolder = tasks.add_parser("regressionFolder")
    regFolder.add_argument("-i", "--input", required=True, help="Input folder")
    regFolder.add_argument("--model", choices=["svm", "knn"],
                           required=True, help="Classifier type")
    regFolder.add_argument("--regression", required=True,
                           help="Regression model to use")

    silrem = tasks.add_parser("silenceRemoval",
                              help="Remove silence segments from a recording")
    silrem.add_argument("-i", "--input", required=True, help="input audio file")
    silrem.add_argument("-s", "--smoothing", type=float, default=1.0,
                        help="smoothing window size in seconds.")
    silrem.add_argument("-w", "--weight", type=float, default=0.5,
                        help="weight factor in (0, 1)")

    spkrDir = tasks.add_parser("speakerDiarization")
    spkrDir.add_argument("-i", "--input", required=True,
                         help="Input audio file")
    spkrDir.add_argument("-n", "--num", type=int, required=True,
                         help="Number of speakers")
    spkrDir.add_argument("--flsd", action="store_true",
                         help="Enable FLsD method")

    speakerDiarizationScriptEval = tasks.add_parser("speakerDiarizationScriptEval",
                                                    help="Train an SVM or KNN "
                                                         "classifier")
    speakerDiarizationScriptEval.add_argument("-i", "--input", required=True,
                                              help="Input directory")
    speakerDiarizationScriptEval.add_argument("--LDAs", type=int, nargs="+",
                                              required=True,
                                              help="List FLsD params")

    thumb = tasks.add_parser("thumbnail",
                             help="Generate a thumbnailWrapper "
                                  "for an audio file")
    thumb.add_argument("-i", "--input", required=True, help="input audio file")
    thumb.add_argument("-s", "--size",  default=10.0,  type=float,
                       help="thumbnailWrapper size in seconds.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    speakerDiarizationWrapper(args.input, args.num)
