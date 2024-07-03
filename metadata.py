name = 'FaceSwapper'
version = '1.0.0-beta'
website = 'https://faceswapper.com'


METADATA =\
{
	'name': 'FaceSwapper',
	'description': '',
	'version': '1.0.0-beta',
	'license': 'MIT',
	'url': 'https://faceswapper.com'
}


def get(key : str) -> str:
	return METADATA[key]