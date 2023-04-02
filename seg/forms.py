from django import forms
from .models import UserUpload

class Imagee(forms.ModelForm):
    class Meta:
        model = UserUpload
        fields = ['image']