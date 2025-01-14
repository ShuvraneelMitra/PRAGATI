import React from "react";
import PdfUploader from "../components/PdfUploader"

const Task1 = () => {

    return (
        <div className="rounded-lg border-titleColor border-[3px]
                        flex justify-between text-white mt-28
                        ml-10 mr-10">
            <div className="p-4 text-[24px] text-justify font-bold">
                The first part of the application takes in a research paper as
                input and via answering various intelligent and context-based questions,
                comes to a conclusion whether the given paper is indeed publishable or not.

                This is done by combining a fact-checker, which searches various reliable
                websites such as ArXiv and Google Scholar to check some facts as they are
                presented in the input paper.
            </div>
            <PdfUploader></PdfUploader>
        </div>
    )
}

export default Task1;