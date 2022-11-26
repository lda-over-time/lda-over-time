Developer's Guide
=================

To create new models for LdaOverTime library, you have to implement a new \
class that inherites and implements all DtmModelInterface's method. Also, your \
model must have the following properties that have yet to be documented on the \
interface:

  - corpus 
  - dates 
  - dafe_format 
  - freq 
  - n_topics 
  - sep 
  - workers 

To build and to publish, we use the `hatch <https://hatch.pypa.io/latest/>` \
package manager. Right bellow, we are listing some commands that are useful, \
feel free to add more examples:
  
  - build: `hatch build`
  - publish (build before): `hatch publish -r "https://upload.pypi.org/legacy/" -u "<username>" -a "<password>"`

lda\_over\_time.models.dtm\_model\_interface
--------------------------------------------

.. automodule:: lda_over_time.models.dtm_model_interface
   :members:
   :undoc-members:
   :show-inheritance:
