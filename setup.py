from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='target_finder_toolkit',
    version='0.1.2',
    author='Ahmed Benakouche',
    author_email='ahmed.benakouche.etudiant@gmail.com',
    description='Widget detection and interaction techniques: Bubble cursor and semantic pointing',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    include_package_data=True,
    package_data={'target_finder_toolkit': ['*.pt'],},
    python_requires='>=3.10',
    install_requires=[
        'numpy',
        'opencv-python',
        'mss',
        'ultralytics',
        'PyQt6',
        'pyautogui',
        'pynput',
    ],
    entry_points={
        'console_scripts': [
            'targetfinder-gui=target_finder_toolkit.targetfinder:main',
            'bubblecursor=target_finder_toolkit.bubblecursor:main',
            'semanticpointing=target_finder_toolkit.semanticpointing:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',
    ],
)

