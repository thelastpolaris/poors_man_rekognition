import tornado.auth
from tornado import escape
import tornado.web
import tornado

class GithubOAuth2Mixin(tornado.auth.OAuth2Mixin):
    """Github authentication using the new Graph API and OAuth2."""
    _OAUTH_ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token?"
    _OAUTH_AUTHORIZE_URL = "https://github.com/login/oauth/authorize?"
    _OAUTH_NO_CALLBACKS = False
    _API_URL = "https://api.github.com"

    async def get_authenticated_user(self, redirect_uri, client_id, client_secret, code, extra_fields=None):
        http = self.get_auth_http_client()
        args = {
            "redirect_uri": redirect_uri,
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
        }

        fields = {'login','id', 'email', 'avatar_url'}
        if extra_fields:
            fields.update(extra_fields)
        response = await http.fetch(
            self._oauth_request_token_url(**args),
            headers={"Accept": "application/json"},
        )
        args = escape.json_decode(response.body)

        session = {
            "access_token": args.get("access_token"),
        }
        assert session["access_token"] is not None

        user = await self.github_request(
            path="/user",
            access_token=session["access_token"],
        )

        if user is None:
            return None

        fieldmap = {}
        for field in fields:
            fieldmap[field] = user.get(field)

        return fieldmap

    async def github_request(self, path, access_token):
        url = self._API_URL + path

        response = await self.get_auth_http_client().fetch(
            url,
            headers={"Authorization": "token {}".format(access_token)},
            user_agent="PMR"
        )

        return escape.json_decode(response.body)
