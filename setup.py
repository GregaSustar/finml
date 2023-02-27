from setuptools import setup, Extension, Command
from Cython.Build import cythonize
import numpy
import os
import shutil


class clean(Command):
    user_options = [("all", "a", "")]

    def initialize_options(self):
        self.all = True
        self._clean_me = []
        self._clean_trees = []
        for root, dirs, files in os.walk("finml"):
            for f in files:
                filepath = os.path.join(root, f)

                if os.path.splitext(f)[-1] in (
                        ".pyc",
                        ".so",
                        ".o",
                        ".pyo",
                        ".pyd",
                        ".c",
                        ".cpp",
                        ".orig",
                        ".html"
                ):
                    self._clean_me.append(filepath)

            for d in ("build", "dist"):
                if os.path.exists(d):
                    self._clean_trees.append(d)

    def finalize_options(self):
        pass

    def run(self):
        for clean_me in self._clean_me:
            try:
                os.unlink(clean_me)
            except OSError:
                pass
        for clean_tree in self._clean_trees:
            try:
                shutil.rmtree(clean_tree)
            except OSError:
                pass

extensions = [
    Extension('finml.data._structures', ['finml/data/_structures.pyx'],
              include_dirs=[numpy.get_include()],
              # libraries=['m'],
              define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')])
]

if __name__ == '__main__':
    setup(
        name='finml',
        ext_modules=cythonize(extensions, language_level="3", annotate=True),
        zip_safe=False,
        cmdclass={'clean': clean},
    )
