import React from "react"

const Description = () => {
    return (
        <div className="flex justify-center rounded-[17px] bg-[#f4c95d]
                        w-3/4 mx-auto mt-28 brightness-300">
            <h1 className="p-10 text-3xl font-extrabold font-custom
                           ">
                Welcome to PRAGATI, an agentic AI application built
                using Pathway which facilitates fact-checking and determines
                the publishability of an uploaded research paper
                based on intelligently generated context-based questions. If the paper
                is publishable, the application also provides suggestions for potential
                top-level conferences where the paper could be submitted.
            </h1>
        </div>
    )
}

export default Description;