@ECHO OFF

PUSHD %~dp0

git clone https://source-xihe-mindspore.osinfra.cn/champagne11/mindcv_twoclass.git
git clone https://source-xihe-mindspore.osinfra.cn/drizzlezyk/ai-real.git

POPD

ECHO Done!
ECHO.

PAUSE
