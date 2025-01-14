import React from "react";
import PdfUploader from "../components/PdfUploader"

const Task2 = () => {
    return (
        <div className="rounded-lg border-titleColor border-[3px]
                        flex justify-between text-white mt-28
                        ml-10 mr-10">
            <div className="p-4 text-[24px] text-justify font-bold">
                Let us assume that the first part of PRAGATI finds a research paper
                to be publishable.

                Then the second part of the application takes in a research paper as
                input and using an elaborate scheme consisting of multiple Reviewer agent
                personas generated via artful prompting, generates a set of conference-
                specific questions that the Answerer agent must reply to. These answers are
                then passed to an Evaluator tool which scores the answers and on the basis of
                the obtained score, determines the most relevant conference out of the given 5:
                EMNLP, CVPR, NIPS, KDD and TMLR for the paper to be submitted to.
            </div>
            <PdfUploader></PdfUploader>
        </div>
    )
}

export default Task2;