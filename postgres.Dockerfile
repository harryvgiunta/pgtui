# The 18.7 source release precedes publication of the official 18.7 image.
# Keep the official entrypoint/runtime, replacing its binaries with the verified release.
FROM postgres:18-alpine@sha256:9ba53c7d41ea093baf5c1ba635ba5434c4f979880527d49bb7d9237d3193085b AS build
ARG PG_VERSION=18.7
ARG PG_SHA256=441dfae4280cae3e0425e5f876f80559fb0731c140f1aae22b3e5a54d0411bfd
RUN apk add --no-cache build-base bison flex perl perl-dev python3-dev tcl-dev curl curl-dev icu-dev openssl-dev zlib-dev lz4-dev zstd-dev linux-headers readline-dev krb5-dev openldap-dev liburing-dev libxml2-dev libxslt-dev llvm21-dev clang21 util-linux-dev
WORKDIR /usr/src/postgresql
RUN curl --fail --location --silent --show-error "https://ftp.postgresql.org/pub/source/v${PG_VERSION}/postgresql-${PG_VERSION}.tar.bz2" -o /tmp/postgresql.tar.bz2 \
    && echo "${PG_SHA256}  /tmp/postgresql.tar.bz2" | sha256sum -c - \
    && tar -xjf /tmp/postgresql.tar.bz2 --strip-components=1 \
    && LLVM_CONFIG=/usr/lib/llvm21/bin/llvm-config CLANG=clang-21 ./configure --prefix=/usr/local --disable-rpath --with-system-tzdata=/usr/share/zoneinfo --with-uuid=e2fs --with-gssapi --with-icu --with-ldap --with-libcurl --with-liburing --with-libxml --with-libxslt --with-llvm --with-lz4 --with-openssl --with-perl --with-python --with-tcl --with-zstd
RUN make -j4 world-bin \
    && make install-world-bin DESTDIR=/opt/postgresql \
    && /opt/postgresql/usr/local/bin/postgres --version

FROM postgres:18-alpine@sha256:9ba53c7d41ea093baf5c1ba635ba5434c4f979880527d49bb7d9237d3193085b
RUN rm -rf /usr/local/lib/postgresql /usr/local/share/postgresql /usr/local/include/postgresql
COPY --from=build /opt/postgresql/usr/local/ /usr/local/
ENV PG_VERSION=18.7
CMD ["postgres", "-c", "listen_addresses=*"]
