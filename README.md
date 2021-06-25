<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
***
***
***
*** To avoid retyping too much info. Do a search and replace for the following:
*** yamaha-bps, cbr_control, twitter_handle, john_mains@yamaha-motor.com, Cyber Control, A handy set of tools for the every day controls engineer.
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]



<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/yamaha-bps/cbr_control">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Cyber Control</h3>

  <p align="center">
    A handy set of tools for the every day controls engineer.
    <br />
    <a href="https://github.com/yamaha-bps/cbr_control/issues">Report Bug</a>
    ·
    <a href="https://github.com/yamaha-bps/cbr_control/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project


### Built With

* [Libboost](https://www.boost.org/)
* [Eigen](https://gitlab.com/libeigen/eigen)
* [yaml-cpp](https://github.com/jbeder/yaml-cpp)
* [GTest](https://github.com/google/googletest)
* [Sophus](https://github.com/strasdat/Sophus)
* [OSQP](https://github.com/osqp/osqp.git)
* [Autodiff](https://github.com/autodiff/autodiff)
* [Cyber Utilities](https://github.com/yamaha-bps/cbr_utils.git)
* [Cyber Math](https://github.com/yamaha-bps/cbr_mat.git)



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* libboost
  ```sh
  sudo apt install libboost-dev
  ```

* Eigen
  ```sh
  sudo apt install libeigen3-dev
  ```

* yaml-cpp
  ```sh
  sudo apt install libyaml-cpp-dev
  ```

* GTest (only necessary to build tests)
  ```sh
  sudo apt install libgtest-dev
  ```

* Sophus
  ```sh
  git clone https://github.com/strasdat/Sophus.git
  mkdir Sophus/build
  cd Sophus/build
  cmake ..
  make -j2
  sudo make install
  ```

* OSQP
  ```sh
  git clone --recursive https://github.com/osqp/osqp.git
  cd osqp
  git checkout v0.6.0
  mkdir build
  cd build
  cmake -G "Unix Makefiles" ..
  cmake --build .
  sudo cmake --build . --target install
  ```

* Autodiff
  ```sh
  git clone https://github.com/autodiff/autodiff.git
  cd autodiff
  git checkout v0.5.12
  mkdir build
  cd build
  cmake ..
  make -j2
  sudo make install
  ```

* Cyber Utilities
  ```sh
  git clone https://github.com/yamaha-bps/cbr_utils.git
  mkdir cbr_utils/build
  cd cbr_utils/build
  cmake .. -DBUILD_TESTING=OFF-DBUILD_EXAMPLES=OFF
  make -j2
  sudo make install
  ```

* Cyber Math
  ```sh
  git clone https://github.com/yamaha-bps/cbr_math.git
  mkdir cbr_math/build
  cd cbr_math/build
  cmake .. -DBUILD_TESTING=OFF-DBUILD_EXAMPLES=OFF
  make -j2
  sudo make install
  ```
### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/yamaha-bps/cbr_control.git
2. Make build directory
   ```sh
   mkdir build
   ```
3. Build
   ```sh
   cd build
   cmake .. -DBUILD_TESTING=ON
   make -j2
   ```
4. Install
   ```sh
   sudo make install
   ```
5. Verify successful install (tests should all pass)
   ```sh
   make test
   ```

6. Uninstall if you don't like it
   ```sh
   sudo make uninstall
   ```


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Ben Mains - john_mains@yamaha-motor.com

Taylor Wentzel - taylor_wentzel@yamaha-motor.com

Project Link: [https://github.com/yamaha-bps/cbr_control](https://github.com/yamaha-bps/cbr_control)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* [Yamaha Motor Corporation](https://yamaha-motor.com/)





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/yamaha-bps/cbr_control.svg?style=for-the-badge
[contributors-url]: https://github.com/yamaha-bps/cbr_control/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/yamaha-bps/cbr_control.svg?style=for-the-badge
[forks-url]: https://github.com/yamaha-bps/cbr_control/network/members
[stars-shield]: https://img.shields.io/github/stars/yamaha-bps/cbr_control.svg?style=for-the-badge
[stars-url]: https://github.com/yamaha-bps/cbr_control/stargazers
[issues-shield]: https://img.shields.io/github/issues/yamaha-bps/cbr_control.svg?style=for-the-badge
[issues-url]: https://github.com/yamaha-bps/cbr_control/issues
[license-shield]: https://img.shields.io/github/license/yamaha-bps/cbr_control.svg?style=for-the-badge
[license-url]: https://github.com/yamaha-bps/cbr_control/blob/master/LICENSE.txt
