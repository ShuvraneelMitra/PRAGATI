import React from 'react'
import Navbar from './components/Navbar'
import Description from './components/Description'
import Task1 from './components/Task1'
import Task2 from './components/Task2'
import team_members from './assets/team_members.png'

function App() {
  return (
    <div>
      <Navbar />
      <Description />
      <img src={team_members} className="mx-auto m-20 w-1000"></img>
      <Task1 />
      <Task2 />
    </div>
  );
}

export default App;
