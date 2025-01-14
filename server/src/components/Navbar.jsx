import React, { useState } from "react";

import github_logo from "../assets/github-mark-white.png";
import mail_logo from "../assets/mail.png"
import info_logo from "../assets/info.png"


const Navbar = () => {

    return (
        <div className='text-white flex justify-between 
                        items-center mx-auto
                        bg-[#999090] bg-opacity-30'>
            <div>
                <h1 className="text-titleColor text-[50px] 
                            font-mono pt-4 pl-4 font-extrabold">
                    PRAGATI
                </h1>
                <h2 className="text-[#ffffff] text-[25px] 
                            font-mono pl-4 pb-4 font-extrabold">
                    Paper Review And Guidance for
                    Academic Target Identification
                </h2>
            </div>
            <ul className="flex">
                <li className="flex p-4 justify-between 
                               font-bold text-[24px] gap-3
                                hover:text-yellow-300">
                    GitHub
                    <img src={github_logo} alt="" className="w-10 h-auto" />
                </li>
                <li className="flex p-4 justify-between 
                               font-bold text-[24px] gap-3
                                hover:text-yellow-300">
                    <a href="mailto: mitrashuvraneel@gmail.com">Contact</a>
                    <img src={mail_logo} alt="" className="w-10 h-auto" />
                </li>
                <li className="flex p-4 justify-between 
                               font-bold text-[24px] gap-3
                                hover:text-yellow-300">
                    About
                    <img src={info_logo} alt="" className="w-10 h-auto" />
                </li>
            </ul>

        </div >
    )
}

export default Navbar;